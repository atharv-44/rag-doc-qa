"""
ingest.py — Step 1-3 of the RAG pipeline

This script reads PDFs, splits them into chunks, generates embeddings,
and stores everything in ChromaDB so we can search it later.

Run this ONCE (or whenever you add new documents):
    python ingest.py
"""

import os
import pdfplumber                          # Extracts text from PDF files
import chromadb                            # Our local vector database
from sentence_transformers import SentenceTransformer  # Free, local embedding model

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DOCUMENTS_DIR = "documents"    # Folder where your PDFs live
CHROMA_DIR    = "chroma_db"    # Folder where ChromaDB will persist data to disk
COLLECTION_NAME = "docs"       # Name of the ChromaDB collection (like a table)

# Why 500 tokens? It's a sweet spot — large enough to hold meaningful context,
# small enough that every chunk stays on-topic. Very long chunks dilute relevance;
# very short chunks lose context.
CHUNK_SIZE    = 500   # Characters per chunk (we approximate tokens with characters here)

# Why overlap? Without it, a sentence split across two chunk boundaries would
# lose its meaning in both halves. 50-char overlap ensures continuity.
OVERLAP       = 50    # Characters shared between consecutive chunks


# ─────────────────────────────────────────────
# STEP 1: EXTRACT TEXT FROM PDFs
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens a PDF and returns all its text as a single string.

    We use pdfplumber because it handles multi-column layouts and tables
    better than simpler libraries — it gives us cleaner text to work with.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:                  # Some pages are images — skip them
                text += page_text + "\n"   # Newline separates pages
    return text


def load_all_documents(docs_dir: str) -> list[dict]:
    """
    Loads every PDF in the documents folder and returns a list of
    {"source": filename, "text": full_text} dicts.
    """
    documents = []
    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️  No PDF files found in '{docs_dir}'. Add some PDFs and re-run.")
        return documents

    for filename in pdf_files:
        path = os.path.join(docs_dir, filename)
        print(f"  📄 Reading: {filename}")
        text = extract_text_from_pdf(path)
        documents.append({"source": filename, "text": text})

    return documents


# ─────────────────────────────────────────────
# STEP 2: SPLIT TEXT INTO CHUNKS
# ─────────────────────────────────────────────

def split_into_chunks(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    """
    Splits a long text into overlapping chunks.

    Why split at all? LLMs have a context window limit. Also, embedding a
    500-char chunk produces a much more precise semantic vector than embedding
    a 50-page document — it makes retrieval far more accurate.

    Returns a list of dicts:
        {"id": unique_id, "text": chunk_text, "source": filename, "chunk_index": n}
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Skip chunks that are mostly whitespace (e.g., blank pages)
        if chunk_text.strip():
            chunks.append({
                "id": f"{source}_chunk_{chunk_index}",   # Unique ID for ChromaDB
                "text": chunk_text,
                "source": source,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        # Move forward by (chunk_size - overlap) so the next chunk
        # starts OVERLAP characters before where this one ended.
        start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────
# STEP 3: EMBED CHUNKS + STORE IN CHROMADB
# ─────────────────────────────────────────────

def embed_and_store(chunks: list[dict], model: SentenceTransformer, collection):
    """
    Generates an embedding vector for each chunk and upserts it into ChromaDB.

    Why embeddings? An embedding converts text into a list of numbers (a vector)
    that captures its *meaning*. Two chunks about the same topic will have vectors
    that point in similar directions — that's what lets us do semantic search.

    Why all-MiniLM-L6-v2? It's tiny (80 MB), fast on CPU, and produces
    high-quality 384-dimensional embeddings. Perfect for local use.
    """
    texts    = [c["text"]   for c in chunks]
    ids      = [c["id"]     for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    print(f"  🔢 Generating embeddings for {len(texts)} chunks…")

    # encode() returns a numpy array of shape (n_chunks, 384)
    # show_progress_bar gives visual feedback for large batches
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # upsert = insert or update (safe to re-run without creating duplicates)
    collection.upsert(
        ids=ids,
        documents=texts,        # Raw text stored alongside the vector
        embeddings=embeddings,
        metadatas=metadatas,    # Extra info we can display as source citations
    )
    print(f"  ✅ Stored {len(chunks)} chunks in ChromaDB.")


# ─────────────────────────────────────────────
# MAIN — wire everything together
# ─────────────────────────────────────────────

def main():
    print("\n🚀 Starting ingestion pipeline…\n")

    # --- Load documents ---
    print("📂 Loading PDFs…")
    documents = load_all_documents(DOCUMENTS_DIR)
    if not documents:
        return

    # --- Split into chunks ---
    print("\n✂️  Splitting text into chunks…")
    all_chunks = []
    for doc in documents:
        chunks = split_into_chunks(doc["text"], doc["source"])
        print(f"  {doc['source']}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    print(f"\n  Total chunks: {len(all_chunks)}")

    # --- Load embedding model ---
    print("\n🤖 Loading embedding model (downloads once, cached after)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # --- Connect to ChromaDB ---
    # PersistentClient saves data to disk so we don't re-embed on every run.
    print(f"\n🗄️  Connecting to ChromaDB at '{CHROMA_DIR}'…")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # get_or_create_collection: idempotent — safe to call multiple times
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # ChromaDB defaults to cosine similarity for text embeddings.
        # Why cosine? It measures the *angle* between vectors, not their length,
        # so it focuses purely on meaning regardless of chunk length.
        metadata={"hnsw:space": "cosine"},
    )

    # --- Embed + store ---
    print("\n📥 Embedding and storing chunks…")
    embed_and_store(all_chunks, model, collection)

    print("\n🎉 Ingestion complete! You can now run: python main.py \"your question\"\n")


if __name__ == "__main__":
    main()
