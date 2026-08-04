# QM2 Beamline Assistant 
(Groq  llama-3.3-70B   ·   LangChain   ·   FastAPI   ·    RAG)

A chatbot for the Quantum Materials beamline (QM2, CHESS ID4B) that answers
questions grounded in the QM2 website
(<https://suchismitasarker.github.io/CHESS-ID4B-QM2/>). It uses **LangChain** to
call Groq's **`llama-3.3-70b-versatile`** model, loads the website as context,
and can run either in the terminal or as a small web API that your GitHub Pages
site can call from the browser.

## Files

```
qm2-chatbot/
├── main.py           # the chatbot (Groq + LangChain; CLI and --serve API modes)
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="gsk_...your key..."     # never commit this
```

## 2. Try it in the terminal

```bash
python qm2_llm.py
```

It loads the QM2 site once, then you chat. The bot answers from the site content
and says when something isn't covered there. Type `exit` to quit.

## 3. Run it as a web API (for the website)

```bash
python main.py             # POST /chat and GET /health on port 8000
```

Quick test:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the QM2 beamline used for?"}'
```

The API keeps a short conversation memory per `session_id` (in memory, cleared on
restart) and only accepts requests from the origins listed in `ALLOWED_ORIGINS`
in `qm2_llm.py` — which already includes your GitHub Pages domain.

## 4. Integrate with the GitHub Pages site

GitHub Pages is **static**, so it can't run Python itself. The flow is: your page
loads a small chat widget in the browser, and that widget calls your running
`qm2_llm.py --serve` backend.

1. Run `python qm2_llm.py --serve` on a machine reachable from the browser (a lab
   server, a small cloud VM, etc.). For anything public you'll want it behind
   HTTPS.
2. Open `chatbot-widget.html`, set `API_BASE` to that backend's URL, and paste the
   whole snippet into your site (for example into `index.html`, or into a Markdown
   page as raw HTML). A floating "QM2 Assistant" button appears bottom-right.
3. Make sure the backend's `ALLOWED_ORIGINS` includes
   `https://suchismitasarker.github.io` (it already does).

### Important: keep the API key on the server

The Groq API key lives only in the backend process (read from `GROQ_API_KEY`). It
is never sent to the browser or embedded in the widget. Don't put the key in the
GitHub Pages site or any client-side code.

## How the website "integration" works

On startup, `load_site_text()` uses LangChain's `WebBaseLoader` to fetch the QM2
page, cleans the text, and injects it into the system prompt as grounding
context. Because the site is a single small page, the whole thing fits in the
prompt — no vector database or embeddings are needed. If the site later grows
into many pages, switch to a retrieval (RAG) setup: split the documents, embed
them (e.g. with a local `sentence-transformers` model, since Groq doesn't provide
embeddings), store them in FAISS, and retrieve the top chunks per question. The
`build_bot()` function is where you'd swap that in.

## Notes

- Model string: `llama-3.3-70b-versatile` (Groq). Change `MODEL` in `qm2_llm.py`
  to use a different Groq model.
- The bot is instructed not to invent facts and to say when the site doesn't
  cover something.
