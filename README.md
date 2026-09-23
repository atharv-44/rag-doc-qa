# RAG-DocQA — Document Q&A with Retrieval-Augmented Generation

A beginner-friendly, minimal RAG pipeline that lets you ask questions about your own PDF documents.  
Everything runs locally except the final LLM call (Groq — free tier, no credit card).

---

## What is RAG?

**Retrieval-Augmented Generation** is a pattern that fixes a core problem with LLMs:  
they only know what they were trained on, and they can hallucinate facts.

RAG solves this by giving the LLM *your documents* as context at query time:

```
Your PDFs → chunks → embeddings → vector DB
                                        ↓
User question → embed → search DB → top-k chunks → LLM → Answer
```

The LLM reads your chunks and answers from them, not from memory.

---

## Project Structure

```
RAG-DocQA/
├── documents/          ← Put your PDF files here (2-3 to start)
├── chroma_db/          ← Auto-created by ingest.py; stores vectors on disk
├── ingest.py           ← Steps 1-3: extract → chunk → embed → store
├── query.py            ← Step 4: embed question → retrieve → LLM answer
├── main.py             ← CLI wrapper (just calls query.py)
├── requirements.txt
├── .env.example        ← Template; copy to .env and add your Groq key
└── README.md
```

---

## Setup

### 1. Prerequisites

- Python 3.10 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First install downloads the `all-MiniLM-L6-v2` model (~80 MB) automatically from HuggingFace.  
> It's cached locally after that — no download on subsequent runs.

### 3. Add your API key

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and replace `your_groq_api_key_here` with your real key.

### 4. Add PDF documents

Drop 2-3 PDF files into the `documents/` folder.  
(Create the folder if it doesn't exist: `mkdir documents`)

### 5. Ingest documents

```bash
python ingest.py
```

This reads your PDFs, splits them into ~500-character chunks with 50-char overlap,
generates embeddings for each chunk, and saves everything to `chroma_db/`.

You only need to re-run this when you add or change documents.

### 6. Ask a question

```bash
python main.py "What are the key findings?"
python main.py "What is the refund policy?"
python main.py "Summarize the methodology section"
```

---

## How Each Part Works

### `ingest.py` — Build the knowledge base

| Step | What happens | Why |
|------|-------------|-----|
| **Extract** | `pdfplumber` reads each page and returns plain text | PDF is binary; we need raw text |
| **Chunk** | Text is split every 500 chars, with 50-char overlap | LLMs have context limits; overlap prevents meaning loss at boundaries |
| **Embed** | `all-MiniLM-L6-v2` converts each chunk → 384-number vector | Vectors capture *meaning*, enabling semantic search |
| **Store** | ChromaDB saves vectors + text to disk | Persistent — no re-embedding on every query |

### `query.py` — Answer a question

| Step | What happens | Why |
|------|-------------|-----|
| **Embed question** | Same model converts question → vector | Must use same model so vectors are comparable |
| **Search** | ChromaDB finds top-3 chunks closest to question vector | Cosine similarity = angle between vectors = semantic closeness |
| **Build prompt** | Retrieved chunks + question → formatted prompt | Grounds the LLM in real content; reduces hallucination |
| **LLM call** | Groq runs `llama-3.1-8b-instant` with your prompt | Reads the context and writes a fluent, grounded answer |

### `main.py` — CLI glue

Parses `sys.argv`, calls `answer_question()`, and prints the answer + source citations.

---

## Configuration Cheatsheet

| Variable | File | Default | Change when… |
|---|---|---|---|
| `CHUNK_SIZE` | `ingest.py` | `500` chars | Answers feel too fragmented (increase) or off-topic (decrease) |
| `OVERLAP` | `ingest.py` | `50` chars | Answers lose context at sentence boundaries (increase) |
| `TOP_K` | `query.py` | `3` | Answers feel incomplete (increase to 5) |
| `LLM_MODEL` | `query.py` | `llama-3.1-8b-instant` | Try `llama-3.1-70b-versatile` for harder questions |
| `temperature` | `query.py` | `0.2` | Higher = more creative; lower = more precise |

---

## Example Output

```
────────────────────────────────────────────────────────────
❓ Question: What is the refund policy?
────────────────────────────────────────────────────────────
🔍 Searching documents…

💬 Answer:
────────────────────────────────────────────────────────────
According to the provided context, customers may request a full
refund within 30 days of purchase, provided the item is unused
and in its original packaging...
────────────────────────────────────────────────────────────

📚 Sources used (top 3):

  [1] File: policy.pdf  |  Chunk #12  |  Similarity: 94.21%
      Preview: "Refund Policy: Customers are entitled to a full refund within 30 days…"

  [2] File: policy.pdf  |  Chunk #13  |  Similarity: 88.73%
      Preview: "Exceptions to the refund policy include digital downloads and…"

  [3] File: faq.pdf  |  Chunk #7  |  Similarity: 79.15%
      Preview: "Q: Can I return a used item? A: Used items are not eligible for…"

────────────────────────────────────────────────────────────
```

---

## Common Errors

| Error | Fix |
|---|---|
| `No PDF files found in 'documents'` | Add PDFs to the `documents/` folder |
| `Collection 'docs' not found` | Run `python ingest.py` first |
| `GROQ_API_KEY not found` | Copy `.env.example` → `.env` and add your key |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |

---

## Limitations (By Design)

This is intentionally a **minimal, educational pipeline**. It does not include:

- **Reranking** — results are ranked purely by cosine similarity
- **Hybrid search** — keyword search (BM25) is not combined with semantic search  
- **Multi-hop retrieval** — only one retrieval step per question
- **Agents** — no tool use or multi-step reasoning

These are well-understood extensions once you understand the core pipeline.
