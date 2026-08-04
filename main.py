import os
import re
import unicodedata
from collections import Counter
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader

# WebBaseLoader warns if no user agent is set.
os.environ.setdefault("USER_AGENT", "qm2-chatbot/1.0")

# Load variables from .env file
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("Missing GROQ_API_KEY in environment or .env file.")

# ── QM2 website integration (multi-page, retrieval-based) ────────────────────
# The bot is grounded in the full QM2 documentation site. The page list is read
# from the "websites" file next to this script (one URL per line); if that file
# is missing, it falls back to the homepage only.
DEFAULT_URLS = ["https://suchismitasarker.github.io/CHESS-ID4B-QM2/"]
MAX_PAGE_CHARS = 40000   # cap of stored FULL text per page (not just the head)
CHUNK_SIZE = 1200        # characters per retrieval passage
CHUNK_OVERLAP = 200      # overlap so tables/sentences aren't split awkwardly
# Smaller context = fewer tokens per request, so the Groq daily token budget
# lasts much longer. The top-scoring passages (e.g. the deadlines table) still
# fit comfortably; raise these only if answers start missing detail.
CONTEXT_BUDGET = 8000    # max chars of context injected into each question
TOP_CHUNKS = 6           # how many best-matching passages to include per question
MAX_HISTORY_MESSAGES = 6 # only replay the last few turns to save tokens
PDF_MAX_CHARS = 200000   # cap of stored text per local PDF file

# Very common words are ignored when scoring so specific terms (e.g. "deadline",
# "proposal", "cryostat") drive retrieval instead of filler like "the"/"what".
STOPWORDS = {
    "the", "and", "for", "are", "was", "with", "that", "this", "you", "your",
    "how", "what", "when", "where", "which", "from", "can", "have", "has",
    "does", "did", "about", "into", "will", "would", "should", "there",
}

# Domain synonym expansion bridges everyday query words to the paper's actual
# vocabulary. Without it, a question about "cooling systems" never matches the
# text/tables that say "cryostream", "cryostat", or "cryocooler". Each query term
# also pulls in its listed synonyms during scoring.
SYNONYMS = {
    "cooling": ["cryo", "cryogenic", "cryostream", "cryostat", "cryocooler", "cryocool", "nitrogen", "helium", "kelvin"],
    "cool": ["cryo", "cryogenic", "cryostream", "cryostat", "cryocooler", "cryocool"],
    "cooled": ["cryo", "cryogenic", "cryostream", "cryostat", "cryocooler"],
    "cooler": ["cryostream", "cryostat", "cryocooler", "cryocool"],
    "cryogenic": ["cryostream", "cryostat", "cryocooler"],
    "temperature": ["cryo", "kelvin", "cryostream", "cryostat", "cryocooler"],
    "thickness": ["thin", "film", "grazing"],
    "film": ["thin", "grazing"],
    "detector": ["pilatus", "eiger", "photon", "counting"],
    "detectors": ["pilatus", "eiger", "photon", "counting"],
    "energy": ["kev", "monochromator", "photon"],
    "flux": ["photon", "photons", "brilliance"],
}


def _load_urls() -> List[str]:
    """Read the documentation page list from the 'websites' file beside this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "websites")
    urls: List[str] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                u = line.strip()
                if u.startswith("http"):
                    urls.append(u)
    except FileNotFoundError:
        pass
    return urls or DEFAULT_URLS


def _clean(text: str) -> str:
    # Normalize Unicode so PDF ligatures (e.g. "ﬁ" -> "fi", "ﬂ" -> "fl") and other
    # compatibility characters become plain ASCII words. Without this, "thin-ﬁlm"
    # tokenizes as "thin"+"lm" and the word "film" is never indexed/searchable.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def load_pages() -> List[dict]:
    """Fetch every QM2 documentation page once and index it for retrieval."""
    urls = _load_urls()
    pages: List[dict] = []
    try:
        docs = WebBaseLoader(urls).load()
    except Exception as exc:  # network problems, etc.
        print(f"[warn] could not load QM2 pages: {exc}")
        return pages
    for d in docs:
        text = _clean(d.page_content)[:MAX_PAGE_CHARS]
        if not text:
            continue
        url = d.metadata.get("source", "")
        pages.append({"url": url, "text": text})
    print(f"[info] loaded {len(pages)} of {len(urls)} QM2 pages for grounding")
    return pages


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split a page into overlapping character windows so any section of the page
    (including tables far below the nav) can be retrieved on its own."""
    chunks: List[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def build_chunks(pages: List[dict]) -> List[dict]:
    """Turn every page into keyword-indexed passages for retrieval."""
    chunks: List[dict] = []
    for p in pages:
        for piece in _chunk_text(p["text"]):
            counts = Counter(re.findall(r"[a-z0-9]{3,}", piece.lower()))
            chunks.append({"url": p["url"], "text": piece, "counts": counts})
    return chunks


def load_pdfs() -> List[dict]:
    """Load any PDF files sitting next to this script into the retrieval corpus.

    Each PDF becomes a 'page' (labelled with its filename) that is then split into
    passages by build_chunks(), so the model can cite and quote from your papers
    exactly the same way it does from the documentation website.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    pages: List[dict] = []
    try:
        from pypdf import PdfReader
    except ImportError:
        print("[warn] pypdf not installed; skipping local PDFs. Run: pip install pypdf")
        return pages
    import glob
    for path in sorted(glob.glob(os.path.join(here, "*.pdf"))):
        name = os.path.basename(path)
        try:
            reader = PdfReader(path)
            raw = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        except Exception as exc:  # unreadable/encrypted/scanned PDF
            print(f"[warn] could not read {name}: {exc}")
            continue
        text = _clean(raw)[:PDF_MAX_CHARS]
        if not text:
            print(f"[warn] no extractable text in {name} (scanned image?) — skipped")
            continue
        pages.append({"url": name, "text": text})
        print(f"[info] loaded PDF {name} ({len(text):,} chars)")
    return pages


# Loaded once at startup and reused for every request.
PAGES = load_pages() + load_pdfs()
CHUNKS = build_chunks(PAGES)
print(f"[info] indexed {len(CHUNKS)} passages across {len(PAGES)} sources")


def retrieve(query: str, top_chunks: int = TOP_CHUNKS, budget: int = CONTEXT_BUDGET) -> str:
    """Return the documentation passages most relevant to the query.

    Passage-level keyword-overlap scoring (no embeddings, since Groq doesn't
    provide them): every page is split into overlapping passages, each passage is
    scored by how often the query's meaningful words appear in it, and the best
    passages are concatenated up to a character budget. Scoring passages rather
    than whole-page heads means a table deep in a long page (e.g. the CHESS
    deadlines) is retrieved on its own instead of being truncated away.
    """
    if not CHUNKS:
        return ("(The QM2 documentation could not be loaded. Answer from general "
                "knowledge of synchrotron beamlines and say when you are unsure.)")
    terms = {t for t in re.findall(r"[a-z0-9]{3,}", query.lower()) if t not in STOPWORDS}
    # Expand with domain synonyms so differently-phrased questions still match.
    for t in list(terms):
        terms.update(SYNONYMS.get(t, ()))
    chosen: List[dict] = []
    if terms:
        scored = []
        for c in CHUNKS:
            score = sum(c["counts"].get(t, 0) for t in terms)
            if score:
                scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = [c for _, c in scored[:top_chunks]]
    if not chosen:  # no keyword hit → fall back to the first few passages
        chosen = CHUNKS[:top_chunks]
    parts, used = [], 0
    for c in chosen:
        block = f"### Source: {c['url']}\n{c['text']}\n"
        if used + len(block) > budget:
            block = block[: max(0, budget - used)]
        if not block:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


SYSTEM_TEMPLATE = """You are the QM2 Assistant, a helpful guide for the \
Quantum Materials beamline (QM2, CHESS ID4B) at the Cornell High Energy \
Synchrotron Source.

Answer questions about the beamline, its capabilities, experiments, data \
analysis, SPEC commands, alignment, detectors, troubleshooting, and how to use \
it. The CONTEXT below contains excerpts from several pages of the QM2 \
documentation, each marked with its source URL. Base your answer on that \
context. If it does not contain the answer, say so plainly and, if helpful, \
suggest which page or who to contact — do not invent facts. When useful, point \
the user to the specific documentation page. Be concise, accurate, and friendly.

--- BEGIN QM2 DOCUMENTATION CONTEXT ---
{site_context}
--- END QM2 DOCUMENTATION CONTEXT ---
"""

app = FastAPI(title="QM2 LangChain Chatbot API", version="1.1")

# Allow your GitHub Pages site (and localhost) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://suchismitasarker.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# 1. Request and Response schemas (unchanged from your version)
class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


# 2. Initialize the LLM
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)

# 3. Prompt template — {site_context} is filled per-question by retrieve()
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Assemble the processing chain
chain = prompt | model

# 4. In-memory session history
# Note: for production, replace with a RedisChatMessageHistory implementation.
store: Dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 5. Wrap the chain with message history management
with_history_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


# ── Built-in chat webpage, served at "/" so you can test in a browser ────────
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QM² Assistant · CHESS ID4B</title>
<style>
  :root{
    --brand1:#4f46e5; --brand2:#0ea5e9; --brand3:#14b8a6;
    --ink:#0b1220; --muted:#64748b; --line:#e6ebf3;
    --panel:#ffffff; --bot:#f6f8fc; --user1:#4f46e5; --user2:#0ea5e9;
    --ok:#22c55e; --shadow:0 24px 70px rgba(15,23,42,.18);
  }
  *{box-sizing:border-box;}
  html,body{height:100%;margin:0;}
  body{
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    color:var(--ink);
    background:
      radial-gradient(900px 500px at 12% -10%,rgba(79,70,229,.20),transparent 60%),
      radial-gradient(900px 520px at 100% 0%,rgba(14,165,233,.18),transparent 55%),
      radial-gradient(800px 600px at 50% 120%,rgba(20,184,166,.16),transparent 55%),
      #eef2f9;
    display:flex;align-items:center;justify-content:center;padding:18px;
  }
  .app{
    width:100%;max-width:780px;height:min(90vh,940px);
    background:var(--panel);border-radius:26px;overflow:hidden;
    box-shadow:var(--shadow);display:flex;flex-direction:column;
    border:1px solid rgba(255,255,255,.7);
  }

  /* header */
  header{
    position:relative;color:#fff;padding:20px 22px;display:flex;align-items:center;gap:14px;
    background:linear-gradient(120deg,var(--brand1),var(--brand2) 60%,var(--brand3));
  }
  header::after{
    content:"";position:absolute;inset:0;pointer-events:none;
    background:radial-gradient(600px 120px at 20% -40%,rgba(255,255,255,.35),transparent);
  }
  .logo{
    width:48px;height:48px;border-radius:15px;flex:none;position:relative;z-index:1;
    display:flex;align-items:center;justify-content:center;font-weight:800;font-size:18px;
    letter-spacing:.5px;background:rgba(255,255,255,.16);
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.35),0 6px 18px rgba(0,0,0,.15);
  }
  .htxt{position:relative;z-index:1;}
  .htxt b{display:block;font-size:17px;font-weight:750;letter-spacing:.2px;}
  .htxt span{display:block;font-size:12.5px;opacity:.9;margin-top:2px;}
  .actions{margin-left:auto;position:relative;z-index:1;display:flex;align-items:center;gap:12px;}
  .status{display:flex;align-items:center;gap:7px;font-size:12.5px;opacity:.95;}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--ok);
    box-shadow:0 0 0 3px rgba(34,197,94,.3);animation:pulse 2s infinite;}
  @keyframes pulse{0%,100%{box-shadow:0 0 0 3px rgba(34,197,94,.3);}50%{box-shadow:0 0 0 6px rgba(34,197,94,.08);}}
  .iconbtn{
    width:34px;height:34px;border-radius:10px;border:0;cursor:pointer;color:#fff;
    background:rgba(255,255,255,.16);display:flex;align-items:center;justify-content:center;transition:.15s;
  }
  .iconbtn:hover{background:rgba(255,255,255,.28);transform:translateY(-1px);}

  /* conversation */
  #log{
    flex:1;overflow-y:auto;padding:24px 22px 8px;display:flex;flex-direction:column;gap:16px;
    scroll-behavior:smooth;
  }
  .row{display:flex;gap:11px;align-items:flex-end;max-width:86%;animation:rise .28s ease both;}
  .row.user{align-self:flex-end;flex-direction:row-reverse;}
  @keyframes rise{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:none;}}
  .av{
    width:32px;height:32px;border-radius:10px;flex:none;display:flex;align-items:center;
    justify-content:center;font-size:12px;font-weight:800;color:#fff;box-shadow:0 3px 10px rgba(15,23,42,.15);
  }
  .row.bot .av{background:linear-gradient(135deg,var(--brand1),var(--brand3));}
  .row.user .av{background:linear-gradient(135deg,#0b1220,#334155);}
  .bubble{
    padding:12px 16px;border-radius:18px;white-space:pre-wrap;word-wrap:break-word;
    line-height:1.55;font-size:14.5px;box-shadow:0 2px 10px rgba(15,23,42,.05);
  }
  .row.bot .bubble{background:var(--bot);border:1px solid var(--line);border-bottom-left-radius:6px;}
  .row.user .bubble{
    background:linear-gradient(135deg,var(--user1),var(--user2));color:#fff;
    border-bottom-right-radius:6px;box-shadow:0 6px 18px rgba(79,70,229,.28);
  }
  .bubble a{color:inherit;text-decoration:underline;text-underline-offset:2px;}
  .row.bot .bubble a{color:var(--brand1);}
  .bubble code{
    font:12.5px/1.4 ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    background:rgba(15,23,42,.06);padding:1px 5px;border-radius:6px;
  }
  .row.user .bubble code{background:rgba(255,255,255,.2);}
  .bubble strong{font-weight:700;}

  .typing{display:flex;gap:5px;padding:3px 2px;}
  .typing i{width:8px;height:8px;border-radius:50%;background:#94a3b8;display:inline-block;
    animation:bounce 1.3s infinite ease-in-out;}
  .typing i:nth-child(2){animation-delay:.16s;}
  .typing i:nth-child(3){animation-delay:.32s;}
  @keyframes bounce{0%,60%,100%{transform:translateY(0);opacity:.5;}30%{transform:translateY(-6px);opacity:1;}}

  /* chips + composer */
  .chips{display:flex;flex-wrap:wrap;gap:9px;padding:10px 22px 4px;}
  .chip{
    border:1px solid var(--line);background:#fff;color:var(--brand1);
    padding:8px 14px;border-radius:999px;font-size:12.5px;cursor:pointer;font-weight:650;transition:.15s;
  }
  .chip:hover{background:#eef2ff;border-color:#c7d2fe;transform:translateY(-1px);box-shadow:0 4px 12px rgba(79,70,229,.14);}
  form{
    display:flex;gap:11px;padding:14px 16px 16px;background:#fff;border-top:1px solid var(--line);
    align-items:flex-end;
  }
  .box{
    flex:1;display:flex;align-items:flex-end;background:#f2f5fb;border:1.5px solid var(--line);
    border-radius:18px;padding:7px 8px 7px 16px;transition:.18s;
  }
  .box:focus-within{border-color:#a5b4fc;background:#fff;box-shadow:0 0 0 4px rgba(79,70,229,.12);}
  textarea{
    flex:1;border:0;background:transparent;resize:none;outline:none;
    font:15px/1.5 inherit;color:var(--ink);padding:7px 0;max-height:140px;
  }
  textarea::placeholder{color:#94a3b8;}
  .send{
    border:0;width:46px;height:46px;flex:none;border-radius:15px;cursor:pointer;color:#fff;
    background:linear-gradient(135deg,var(--brand1),var(--brand2));
    display:flex;align-items:center;justify-content:center;transition:.15s;
    box-shadow:0 8px 20px rgba(79,70,229,.32);
  }
  .send:hover{filter:brightness(1.08);transform:translateY(-1px);}
  .send:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none;}
  .foot{text-align:center;font-size:11.5px;color:var(--muted);padding:0 0 12px;background:#fff;}
  .foot b{color:var(--brand1);font-weight:700;}

  #log::-webkit-scrollbar{width:10px;}
  #log::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:10px;border:3px solid #fff;}
  #log::-webkit-scrollbar-thumb:hover{background:#94a3b8;}
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="logo">QM²</div>
    <div class="htxt">
      <b>QM² Assistant</b>
      <span>Quantum Materials beamline · CHESS ID4B</span>
    </div>
    <div class="actions">
      <div class="status"><span class="dot"></span> online</div>
      <button class="iconbtn" id="clear" type="button" title="New chat" aria-label="New chat">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4h8v2"/>
          <path d="M6 6l1 14h10l1-14"/></svg>
      </button>
    </div>
  </header>

  <div id="log"></div>

  <div class="chips" id="chips">
    <div class="chip">What is the QM2 beamline?</div>
    <div class="chip">When is the proposal deadline?</div>
    <div class="chip">How do I analyze my data?</div>
    <div class="chip">What detectors are available?</div>
  </div>

  <form id="f">
    <div class="box">
      <textarea id="i" rows="1" placeholder="Ask about the QM2 beamline…" autofocus></textarea>
    </div>
    <button class="send" id="send" type="submit" aria-label="Send">
      <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg>
    </button>
  </form>
  <div class="foot">Grounded in the <b>QM² documentation</b> · answers may be imperfect — verify critical steps</div>
</div>

<script>
  let sessionId = "web-" + Math.random().toString(36).slice(2);
  const log = document.getElementById("log");
  const f = document.getElementById("f");
  const i = document.getElementById("i");
  const send = document.getElementById("send");
  let chips = document.getElementById("chips");

  function fmt(t){
    let s = t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    s = s.replace(/\*\*(.+?)\*\*/g,"<strong>$1</strong>");
    s = s.replace(/`([^`]+?)`/g,"<code>$1</code>");
    s = s.replace(/(https?:\/\/[^\s<]+)/g,'<a href="$1" target="_blank" rel="noopener">$1</a>');
    return s;
  }

  function bubble(text, who, raw){
    const row = document.createElement("div");
    row.className = "row " + who;
    const av = document.createElement("div");
    av.className = "av";
    av.textContent = who === "user" ? "You" : "Q";
    const b = document.createElement("div");
    b.className = "bubble";
    if(raw){ b.innerHTML = fmt(text); } else { b.textContent = text; }
    row.appendChild(av); row.appendChild(b);
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
    return b;
  }

  function typing(){
    const row = document.createElement("div");
    row.className = "row bot";
    row.innerHTML = '<div class="av">Q</div><div class="bubble"><div class="typing"><i></i><i></i><i></i></div></div>';
    log.appendChild(row);
    log.scrollTop = log.scrollHeight;
    return row;
  }

  function greet(){
    bubble("Hi! I'm the QM² Assistant. Ask me anything about the Quantum Materials beamline at CHESS ID4B — capabilities, SPEC commands, alignment, detectors, data analysis, or troubleshooting.", "bot");
  }
  greet();

  i.addEventListener("input", () => {
    i.style.height = "auto";
    i.style.height = Math.min(i.scrollHeight, 140) + "px";
  });
  i.addEventListener("keydown", (e) => {
    if(e.key === "Enter" && !e.shiftKey){ e.preventDefault(); f.requestSubmit(); }
  });

  async function sendMsg(text){
    const msg = (text || i.value).trim();
    if(!msg) return;
    i.value = ""; i.style.height = "auto";
    if(chips){ chips.remove(); chips = null; }
    bubble(msg, "user");
    send.disabled = true;
    const t = typing();
    try{
      const r = await fetch("/chat", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({message:msg, session_id:sessionId})
      });
      const d = await r.json();
      t.remove();
      bubble(d.response || d.detail || "(no reply)", "bot", true);
    }catch(err){
      t.remove();
      bubble("⚠️ Couldn't reach the backend. Is the server running?", "bot");
    }finally{
      send.disabled = false; i.focus();
    }
  }

  f.addEventListener("submit", (e) => { e.preventDefault(); sendMsg(); });
  document.getElementById("chips").addEventListener("click", (e) => {
    if(e.target.classList.contains("chip")) sendMsg(e.target.textContent);
  });
  document.getElementById("clear").addEventListener("click", () => {
    log.innerHTML = "";
    sessionId = "web-" + Math.random().toString(36).slice(2);
    greet();
    i.focus();
  });
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    """Chat webpage — open http://<host>:<port>/ in a browser."""
    return INDEX_HTML


@app.get("/health")
def health():
    pdfs = [p["url"] for p in PAGES if p["url"].lower().endswith(".pdf")]
    return {
        "status": "ok",
        "sources_loaded": len(PAGES),
        "web_pages": len(PAGES) - len(pdfs),
        "pdf_sources": pdfs,          # confirm your PDF is in the index here
        "passages_indexed": len(CHUNKS),
        "total_chars": sum(len(p["text"]) for p in PAGES),
    }


# 6. Chat endpoint (your schema, grounded in the most relevant QM2 pages)
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        context = retrieve(request.message)  # pick the most relevant pages
        # Keep only the most recent turns so long chats don't inflate token use.
        history = get_session_history(request.session_id)
        if len(history.messages) > MAX_HISTORY_MESSAGES:
            history.messages[:] = history.messages[-MAX_HISTORY_MESSAGES:]
        config = {"configurable": {"session_id": request.session_id}}
        result = await with_history_chain.ainvoke(
            {"input": request.message, "site_context": context},
            config=config,
        )
        return ChatResponse(response=result.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Single stable process: loads the 41 pages once, no NFS file-watching.
    # (reload=True is a dev convenience that re-fetches every page on each reload
    # and watches the filesystem — too heavy for a login node like lnx201.)
    # Passing the app object avoids any dependence on the filename.
    # host=0.0.0.0 so an SSH tunnel from your laptop can reach it.
    uvicorn.run(app, host="0.0.0.0", port=8000)
