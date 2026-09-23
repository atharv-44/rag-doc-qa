
"""
query.py — Step 4 of the RAG pipeline

Given a user question, this module:
  1. Embeds the question using the same model used during ingestion
  2. Searches ChromaDB for the top-k most semantically similar chunks
  3. Sends those chunks + the question to an LLM to generate an answer

This is the "RA" part of RAG.
"""

import os

# Disable ChromaDB anonymous telemetry.
# This must be set BEFORE importing chromadb.
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv


# Load GROQ_API_KEY from .env
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "docs"

# Must match the model used during ingestion.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of chunks to retrieve.
TOP_K = 3

# Working Groq model.
LLM_MODEL = "openai/gpt-oss-20b"


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(
    question: str,
    collection,
    model: SentenceTransformer
) -> dict:
    """Embed the question and retrieve relevant chunks."""

    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ],
    )

    return results


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_prompt(
    question: str,
    context_chunks: list[str]
) -> str:
    """Build the prompt using retrieved document context."""

    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

If the answer is not present in the context, say:
"I don't have enough information to answer that."

Do not invent information or use unrelated knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    return prompt


# ============================================================
# GENERATION
# ============================================================

def ask_llm(prompt: str) -> str:
    """Send the prompt to Groq and return the generated answer."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found.\n"
            "Make sure your .env file contains:\n"
            "GROQ_API_KEY=your_api_key"
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise document assistant. "
                    "Answer only using the provided document context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=512,
    )

    return response.choices[0].message.content


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def answer_question(question: str) -> dict:
    """
    Complete RAG pipeline:

    Question
        ↓
    Embedding
        ↓
    ChromaDB retrieval
        ↓
    Context construction
        ↓
    Groq LLM
        ↓
    Answer + sources
    """

    # Load embedding model.
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to existing ChromaDB.
    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    # Get existing collection.
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found "
            f"in '{CHROMA_DIR}'.\n"
            "Run 'python ingest.py' first to build "
            "the vector store."
        )

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    results = retrieve_relevant_chunks(
        question,
        collection,
        model
    )

    retrieved_texts = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]
    retrieved_distances = results["distances"][0]

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    prompt = build_prompt(
        question,
        retrieved_texts
    )

    answer = ask_llm(prompt)

    # --------------------------------------------------------
    # Prepare source information
    # --------------------------------------------------------

    sources = [
        {
            "text": text,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "similarity": round(
                1 - dist,
                4
            ),
        }
        for text, meta, dist in zip(
            retrieved_texts,
            retrieved_metadatas,
            retrieved_distances
        )
    ]

    return {
        "answer": answer,
        "sources": sources
    }
