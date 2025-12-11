import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

clean_prompt = ChatPromptTemplate.from_template("""
Extract the latest user message and assistant response from this chat log.

Return valid JSON only in the following format:

{{
  "user_query": "...",
  "assistant_answer": "..."
}}

Chat Log:
{raw_json}
""")


def extract_clean_chat(raw_text: str):
    # Load GROQ API key at runtime
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key.strip() == "":
        raise ValueError("GROQ_API_KEY is not set. Please enter it in Streamlit sidebar.")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key
    )

    prompt = clean_prompt.format(raw_json=raw_text)
    response = llm.invoke(prompt)

    # response.content contains JSON text
    import json
    try:
        return json.loads(response.content)
    except:
        return {"user_query": "", "assistant_answer": ""}
