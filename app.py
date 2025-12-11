import streamlit as st
import json
import time
import os

from json_cleaner import extract_clean_chat
from utils.embeddings import load_model, embed_text, embed_texts
from utils.scoring import relevance_score, completeness_score, hallucination_score


# ==========================================================
# Helper: extract context texts
# ==========================================================
def extract_context_texts(context_json):
    if "results" in context_json:
        return [c.get("text", "") for c in context_json["results"] if isinstance(c, dict)]

    if "data" in context_json and "vector_data" in context_json["data"]:
        return [
            c.get("text", "")
            for c in context_json["data"]["vector_data"]
            if isinstance(c, dict) and "text" in c
        ]

    texts = []
    def recurse(obj):
        if isinstance(obj, dict):
            if "text" in obj and isinstance(obj["text"], str):
                texts.append(obj["text"])
            for v in obj.values():
                recurse(v)
        elif isinstance(obj, list):
            for v in obj:
                recurse(v)
    recurse(context_json)
    return texts


def estimate_tokens(text):
    return len(text.split())


# ==========================================================
# Streamlit UI Settings
# ==========================================================
st.set_page_config(page_title="LLM Evaluation Pipeline", layout="wide")

# Title Section
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        🧠 LLM Evaluation Pipeline  
    </h1>
    <p style='text-align: center; font-size:18px; color: #555;'>
        Upload your chat and context JSON files to evaluate relevance, completeness, and hallucinations.
    </p>
    <br>
    """,
    unsafe_allow_html=True
)

# ---------------- Sidebar ----------------
st.sidebar.title("🔐 API Settings")
groq_key = st.sidebar.text_input("Enter GROQ API Key", type="password", help="Required to clean chat.json using LLM")

if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key


# ---------------- Main Screen ----------------
st.markdown("### 📤 Upload Files")

colA, colB = st.columns(2)

with colA:
    chat_file = st.file_uploader("📄 Upload chat.json", type=["json"])

with colB:
    context_file = st.file_uploader("📚 Upload context.json", type=["json"])

run_button = st.button("🚀 Run Evaluation", use_container_width=True)

# ==========================================================
# Run Evaluation
# ==========================================================
if run_button:

    if not groq_key:
        st.error("❌ Please enter your GROQ API key in the sidebar.")
        st.stop()

    if not chat_file or not context_file:
        st.error("❌ Please upload BOTH chat.json and context.json.")
        st.stop()

    # ---------------- CLEAN chat.json ----------------
    chat_raw = chat_file.read().decode("utf-8")

    with st.spinner("🤖 Cleaning chat.json using LLM..."):
        cleaned = extract_clean_chat(chat_raw)

    user_query = cleaned.get("user_query", "")
    llm_answer = cleaned.get("assistant_answer", "")

    # ---------------- Load context.json ----------------
    context_json = json.load(context_file)
    context_texts = extract_context_texts(context_json)

    if not context_texts:
        st.error("❌ No 'text' fields found in context.json.")
        st.stop()

    # ---------------- Embedding + Scoring ----------------
    with st.spinner("⚙️ Running evaluation..."):
        model = load_model()

        start = time.time()

        user_emb = embed_text(user_query, model)
        answer_emb = embed_text(llm_answer, model)
        context_embs = embed_texts(context_texts, model)

        combined_ctx = " ".join(context_texts)
        combined_ctx_emb = embed_text(combined_ctx, model)

        relevance = relevance_score(user_emb, answer_emb)
        completeness = completeness_score(answer_emb, combined_ctx_emb)
        hallucination = hallucination_score(answer_emb, context_embs)

        latency_ms = (time.time() - start) * 1000
        tokens = estimate_tokens(llm_answer)
        cost = tokens * 0.000001

    # ---------------- DISPLAY RESULTS ----------------

    st.success("🎉 Evaluation Completed!")

    st.markdown("### 📝 Extracted Messages")
    st.write(f"**👤 User Query:**  \n{user_query}")
    st.write(f"**🤖 LLM Answer:**  \n{llm_answer}")

    # ---------------- Evaluation Metrics ----------------
    st.markdown("### 📊 Evaluation Metrics (Scores 0 → 1)")

    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 Relevance", round(relevance, 4))
    m2.metric("📘 Completeness", round(completeness, 4))
    m3.metric("⚠️ Hallucination", round(hallucination, 4))

    m4, m5 = st.columns(2)
    m4.metric("⏱️ Latency (ms)", round(latency_ms, 2))
    m5.metric("💰 Estimated Cost (USD)", round(cost, 6))

    # ---------------- Context Viewer ----------------
    st.markdown("### 📚 Context Chunks")
    with st.expander("🔍 View Extracted Context Texts"):
        for i, txt in enumerate(context_texts):
            st.markdown(f"#### 🔸 Chunk {i+1}")
            st.write(txt)
            st.write("---")
