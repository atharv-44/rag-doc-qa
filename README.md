# 🔍 RAG-DocQA — AI Document Q&A System

> Ask questions about your own PDF documents using a fully local RAG (Retrieval-Augmented Generation) pipeline.  
> No OpenAI. No cloud embeddings. No black boxes — every part is explained.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://www.trychroma.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq%20(free)-green)](https://console.groq.com)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION  (run once)                   │
│                                                             │
│   PDF Files  ──►  Extract Text  ──►  Split into Chunks      │
│                                           │                 │
│                                     Embed (MiniLM)          │
│                                           │                 │
│                                     ChromaDB (disk)         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     QUERY  (every question)                 │
│                                                             │
│   User Question  ──►  Embed Question                        │
│                              │                              │
│                        ChromaDB Search                      │
│                        (cosine similarity)                  │
│                              │                              │
│                        Top-3 Chunks                         │
│                              │                              │
│                     Build Prompt (chunks + question)        │
│                              │                              │
│                       Groq LLM API                          │
│                       (llama-3.1-8b)                        │
│                              │                              │
│                    Answer + Source Citations ◄──────────────┤
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📄 PDF ingestion | Extracts and chunks text from multiple PDFs |
| 🧠 Local embeddings | `all-MiniLM-L6-v2` — runs on CPU, no API key |
| 🗄️ Vector storage | ChromaDB — persistent on disk, no server needed |
| 🤖 LLM generation | Groq's free API (llama-3.1-8b-instant) |
| 🌐 Web UI | Streamlit app with source citations |
| 💻 CLI | `python main.py "question"` for terminal use |
| 🔒 Private | Your documents never leave your machine (only the final prompt goes to Groq) |

---

## 📁 Project Structure

```
RAG-DocQA/
├── app.py              ← 🌐 Streamlit web UI  (NEW)
├── ingest.py           ← Step 1-3: PDF → chunks → embeddings → ChromaDB
├── query.py            ← Step 4: embed question → retrieve → LLM answer
├── main.py             ← CLI: python main.py "your question"
├── requirements.txt    ← All dependencies
├── .env.example        ← API key template
├── .gitignore          ← Keeps .env and chroma_db/ off GitHub
├── documents/          ← Put your PDFs here
└── chroma_db/          ← Auto-created by ingest.py (don't commit this)
```

---

## 🚀 Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Open `.env` and paste your free Groq API key from [console.groq.com](https://console.groq.com).

### 3. Add PDF documents

Drop your PDF files into the `documents/` folder.

### 4. Ingest documents *(run once)*

```bash
python ingest.py
```

This reads PDFs → splits into chunks → generates embeddings → saves to ChromaDB.  
Only needs to be re-run when you add new documents.

### 5a. Launch Web UI *(recommended)*

```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`

### 5b. Use CLI *(alternative)*

```bash
python main.py "What are the key topics in module 1?"
```

---

## 🧠 How RAG Works — Explained Simply

**Problem:** LLMs only know what they were trained on. They can't answer questions about *your* documents.

**Solution:** RAG = give the LLM your documents as context at the time of the question.

```
Without RAG:  Question ──► LLM ──► Answer (from training data, may hallucinate)

With RAG:     Question ──► Search Docs ──► Relevant Chunks ──► LLM ──► Grounded Answer
```

The LLM acts as a **reader**, not a **memorizer** — it reads your chunks and answers from them.

---

## ⚙️ Key Design Decisions

| Decision | Reasoning |
|---|---|
| **500-char chunks** | Large enough for context, small enough for precise retrieval |
| **50-char overlap** | Prevents meaning loss when sentences straddle chunk boundaries |
| **Cosine similarity** | Measures semantic angle between vectors, independent of text length |
| **Top-K = 3** | Enough context without overloading the LLM's window |
| **all-MiniLM-L6-v2** | 80 MB, CPU-friendly, 384-dim vectors, fully free |
| **temperature = 0.2** | Low creativity = factual, grounded answers |

---

## 🛠️ Configuration

| Variable | File | Default | Adjust when… |
|---|---|---|---|
| `CHUNK_SIZE` | `ingest.py` | `500` | Answers too fragmented → increase |
| `OVERLAP` | `ingest.py` | `50` | Context lost at boundaries → increase |
| `TOP_K` | `query.py` | `3` | Answers incomplete → increase to 5 |
| `LLM_MODEL` | `query.py` | `llama-3.1-8b-instant` | Need higher quality → use `llama-3.1-70b-versatile` |
| `temperature` | `query.py` | `0.2` | Need creative answers → increase |

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `No PDF files found` | Add PDFs to `documents/` folder |
| `Collection 'docs' not found` | Run `python ingest.py` first |
| `GROQ_API_KEY not found` | Copy `.env.example` → `.env` and add your key |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |

---

## 🗺️ What This Doesn't Include *(by design)*

This is an intentionally clean, educational pipeline. It does not include reranking, hybrid search, agents, or multi-hop retrieval — these are well-understood extensions once you understand the core loop.

---

## 📄 License

MIT — free to use, modify, and learn from.
