"""
main.py — CLI entry point

Usage:
    python main.py "What is the refund policy?"
    python main.py "Summarize the key findings."

This file is intentionally thin — it only handles user input/output.
All the real logic lives in query.py (and ingest.py for setup).
"""

import sys
from query import answer_question


def print_separator(char="─", width=60):
    print(char * width)


def main():
    # ── Validate input ──
    if len(sys.argv) < 2:
        print("Usage: python main.py \"your question here\"")
        print("Example: python main.py \"What are the main findings?\"")
        sys.exit(1)

    # Join all args after the script name so the user doesn't NEED quotes
    # e.g., python main.py What is RAG?  →  works fine
    question = " ".join(sys.argv[1:])

    print_separator()
    print(f"❓ Question: {question}")
    print_separator()
    print("🔍 Searching documents…\n")

    # ── Run the RAG pipeline ──
    try:
        result = answer_question(question)
    except RuntimeError as e:
        # Friendly error if the user forgot to run ingest.py first
        print(f"❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        # Friendly error if GROQ_API_KEY is missing
        print(f"❌ Config error: {e}")
        sys.exit(1)

    # ── Print the answer ──
    print("💬 Answer:")
    print_separator()
    print(result["answer"])
    print_separator()

    # ── Print sources so the user can verify where the answer came from ──
    # Showing sources is essential for trust — RAG answers are only as good
    # as the retrieved chunks, so users should always be able to audit them.
    print(f"\n📚 Sources used (top {len(result['sources'])}):\n")
    for i, source in enumerate(result["sources"], start=1):
        print(f"  [{i}] File: {source['source']}  |  Chunk #{source['chunk_index']}  |  Similarity: {source['similarity']:.2%}")
        # Print a preview of the chunk so the user can see exactly what text
        # the answer was based on (truncated to keep the terminal readable)
        preview = source["text"].replace("\n", " ").strip()[:200]
        print(f"      Preview: \"{preview}…\"\n")

    print_separator()


if __name__ == "__main__":
    main()
