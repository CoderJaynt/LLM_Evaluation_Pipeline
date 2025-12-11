import json
import time
import argparse

from json_cleaner import extract_clean_chat
from utils.embeddings import load_model, embed_text, embed_texts
from utils.scoring import relevance_score, completeness_score, hallucination_score


def extract_context_texts(context_json):
    """
    Extract text chunks from ANY context.json structure.
    Supports:
    - context["results"]
    - context["data"]["vector_data"]
    - fallback: find all 'text' fields anywhere
    """

    # Case 1 — standard expected format
    if "results" in context_json:
        return [c.get("text", "") for c in context_json["results"] if isinstance(c, dict)]

    # Case 2 — YOUR actual format: data -> vector_data
    if "data" in context_json and "vector_data" in context_json["data"]:
        return [
            c.get("text", "")
            for c in context_json["data"]["vector_data"]
            if isinstance(c, dict) and "text" in c
        ]

    # Case 3 — fallback: search recursively
    texts = []

    def recurse(obj):
        if isinstance(obj, dict):
            if "text" in obj and isinstance(obj["text"], str):
                texts.append(obj["text"])
            for value in obj.values():
                recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)

    recurse(context_json)

    return texts


def estimate_tokens(text: str) -> int:
    """Rough token count based on whitespace split."""
    return len(text.split())


def evaluate_pipeline(chat_raw_text: str, context_json: dict):
    """
    Main evaluation function.
    Args:
        chat_raw_text: raw (possibly invalid) chat.json text
        context_json: parsed context.json dictionary
    """

    # ------------------------------------------------------------
    # STEP 1 — CLEAN CHAT.JSON USING LLM (Extract only needed data)
    # ------------------------------------------------------------
    cleaned = extract_clean_chat(chat_raw_text)

    user_query = cleaned.get("user_query", "")
    llm_answer = cleaned.get("assistant_answer", "")

    context_texts = extract_context_texts(context_json)

    if not context_texts:
         raise ValueError("Could not find any 'text' fields in context.json.")


    # ------------------------------------------------------------
    # STEP 3 — LOAD EMBEDDING MODEL
    # ------------------------------------------------------------
    model = load_model()

    start = time.time()

    # ------------------------------------------------------------
    # STEP 4 — EMBEDDINGS
    # ------------------------------------------------------------
    user_emb = embed_text(user_query, model)
    answer_emb = embed_text(llm_answer, model)
    context_embs = embed_texts(context_texts, model)

    context_combined = " ".join(context_texts)
    context_combined_emb = embed_text(context_combined, model)

    # ------------------------------------------------------------
    # STEP 5 — SCORES
    # ------------------------------------------------------------
    relevance = relevance_score(user_emb, answer_emb)
    completeness = completeness_score(answer_emb, context_combined_emb)
    hallucination = hallucination_score(answer_emb, context_embs)

    latency_ms = (time.time() - start) * 1000
    tokens = estimate_tokens(llm_answer)
    cost = tokens * 0.000001

    # ------------------------------------------------------------
    # STEP 6 — RETURN REPORT
    # ------------------------------------------------------------
    return {
        "user_query": user_query,
        "llm_answer": llm_answer,

        "relevance_score": round(relevance, 4),
        "completeness_score": round(completeness, 4),
        "hallucination_score": round(hallucination, 4),

        "latency_ms": round(latency_ms, 2),
        "token_estimate": tokens,
        "cost_estimate_usd": round(cost, 6)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", required=True, help="Path to raw chat.json")
    parser.add_argument("--context", required=True, help="Path to context.json")
    args = parser.parse_args()

    # Load raw chat.json (not parsed)
    with open(args.chat, "r", encoding="utf-8") as f:
        chat_raw = f.read()

    # Load clean context.json
    with open(args.context, "r", encoding="utf-8") as f:
        context_json = json.load(f)

    result = evaluate_pipeline(chat_raw, context_json)

    print("\n=== Evaluation Report ===")
    print(json.dumps(result, indent=4))
