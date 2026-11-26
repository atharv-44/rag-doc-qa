"""
app.py — Streamlit Web UI for RAG-DocQA

This file wraps the same query.py pipeline in a browser-based interface.
Streamlit lets you build web apps purely in Python — no HTML/CSS/JS needed.

Run with:
    streamlit run app.py

How it works:
  - You type a question in the browser
  - It calls the same answer_question() from query.py
  - The answer + sources are displayed in a clean layout
"""

import os
import streamlit as st
from query import answer_question   # Reuses our existing RAG pipeline — no duplication

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  — must be the FIRST Streamlit call in the file
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🔍",
    layout="wide",           # Use full browser width for better readability
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — minimal styling to look polished without overcomplicating
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Make the answer box stand out */
    .answer-box {
        background-color: #f0f7ff;
        border-left: 4px solid #2563eb;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-size: 1rem;
        line-height: 1.6;
        color: #1e293b;
    }
    /* Source card styling */
    .source-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: #475569;
    }
    .similarity-high   { color: #16a34a; font-weight: bold; }  /* green  */
    .similarity-medium { color: #d97706; font-weight: bold; }  /* amber  */
    .similarity-low    { color: #dc2626; font-weight: bold; }  /* red    */
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  — project info + how-to-use guide
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/document.png", width=60)
    st.title("RAG Doc Q&A")
    st.caption("Ask questions about your PDF documents using AI.")

    st.divider()

    st.markdown("### ⚙️ How it works")
    st.markdown("""
    1. **Ingest** — PDFs are split into chunks & embedded  
    2. **Retrieve** — Your question is matched to the top 3 most relevant chunks  
    3. **Generate** — An LLM reads those chunks and answers your question  
    """)

    st.divider()

    st.markdown("### 📋 Setup")
    st.code("python ingest.py", language="bash")
    st.caption("Run this once to index your documents before asking questions.")

    st.divider()

    # Let users adjust Top-K from the sidebar without touching code
    # Why expose this? It's a great teaching moment — interviewers love it
    top_k = st.slider(
        "Top-K chunks to retrieve",
        min_value=1,
        max_value=6,
        value=3,
        help="How many document chunks to pass to the LLM as context. "
             "More = richer context but slower. Usually 3 is ideal."
    )

    st.divider()
    st.caption("Built with: Python · ChromaDB · SentenceTransformers · Groq · Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔍 Document Q&A — RAG Pipeline")
st.markdown(
    "Ask anything about your ingested PDFs. "
    "The AI will search your documents and generate a grounded answer."
)

st.divider()

# ── Question input ──────────────────────────────────────────────────────────

question = st.text_input(
    label="Your question",
    placeholder='e.g. "What are the key topics covered in module 1?"',
    label_visibility="collapsed",
)

ask_btn = st.button("🔎 Ask", type="primary", use_container_width=False)

# ─────────────────────────────────────────────────────────────────────────────
# HANDLE QUERY
# ─────────────────────────────────────────────────────────────────────────────

if ask_btn and question.strip():

    # st.spinner gives visual feedback while the pipeline runs (~3-8 seconds)
    with st.spinner("Searching documents and generating answer…"):
        try:
            # Pass top_k from the sidebar slider to the pipeline
            # We temporarily monkey-patch the module-level constant so the
            # existing query.py doesn't need any changes
            import query as _query_module
            original_top_k = _query_module.TOP_K
            _query_module.TOP_K = top_k

            result = answer_question(question)

            _query_module.TOP_K = original_top_k   # Restore original value

        except RuntimeError as e:
            st.error(f"⚠️ {e}")
            st.info("Run `python ingest.py` first to build the vector store.")
            st.stop()

        except ValueError as e:
            st.error(f"🔑 API key error: {e}")
            st.info("Add your GROQ_API_KEY to a `.env` file in the project folder.")
            st.stop()

    # ── Answer ───────────────────────────────────────────────────────────────

    st.markdown("### 💬 Answer")
    st.markdown(
        f'<div class="answer-box">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Sources ──────────────────────────────────────────────────────────────
    # Showing sources is what makes RAG trustworthy — users can verify every answer.

    st.markdown(f"### 📚 Sources used ({len(result['sources'])} chunks)")
    st.caption(
        "These are the exact document sections the answer was based on. "
        "Similarity score = how closely the chunk matched your question."
    )

    for i, source in enumerate(result["sources"], start=1):
        sim = source["similarity"]

        # Color-code the similarity score so it's instantly readable
        if sim >= 0.75:
            sim_class = "similarity-high"
            sim_label = "High"
        elif sim >= 0.50:
            sim_class = "similarity-medium"
            sim_label = "Medium"
        else:
            sim_class = "similarity-low"
            sim_label = "Low"

        # Use Streamlit expanders so the UI stays clean; users click to read full chunk
        with st.expander(
            f"[{i}]  📄 {source['source']}  —  Chunk #{source['chunk_index']}  "
            f"  |  Match: {sim:.0%}  ({sim_label})",
            expanded=(i == 1),   # Auto-expand only the top source
        ):
            st.markdown(f"**Similarity score:** `{sim:.4f}`")
            st.markdown("**Chunk text:**")
            # Show the raw chunk text the LLM actually read
            st.text(source["text"])

elif ask_btn and not question.strip():
    st.warning("Please type a question before clicking Ask.")

# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE  — shown before the user asks anything
# ─────────────────────────────────────────────────────────────────────────────

else:
    if not (ask_btn and question.strip()):
        st.markdown("#### 💡 Example questions to try:")
        examples = [
            "What are the key topics covered in the documents?",
            "Summarize the main concepts.",
            "What is explained in module 1?",
            "What algorithms are discussed?",
        ]
        # Display example questions as clickable pills
        cols = st.columns(2)
        for idx, ex in enumerate(examples):
            with cols[idx % 2]:
                st.info(f"❝ {ex}")
