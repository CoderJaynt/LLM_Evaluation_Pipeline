import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# Create LLM
llm = ChatGroq(model="llama-3.1-8b-instant",api_key="")  ## *******PUT YOUR GROQ API KEY HERE******


# Prompt to clean and normalize messy chat JSON
clean_prompt = ChatPromptTemplate.from_template("""
You are a JSON cleaning assistant.

Given the following raw chat data (may be invalid JSON, partial JSON, or messy text):

{raw_json}

Extract ONLY the following:

1. The latest **User** message
2. The latest **Assistant / AI / Chatbot** message

Return a **valid JSON** object in exactly this format:

{{
  "user_query": "...",
  "assistant_answer": "..."
}}

Make sure to:
- ALWAYS return valid JSON
- NEVER include comments or explanations
- If something is missing, return an empty string for that field
""")




def extract_clean_chat(raw_text: str) -> dict:
    """
    Pass raw text to LLM → get clean JSON structure.

    Returns:
        {
          "user_query": "...",
          "assistant_answer": "..."
        }
    """
    prompt = clean_prompt.format(raw_json=raw_text)
    response = llm.invoke(prompt)

    try:
        cleaned = json.loads(response.content)
    except Exception:
        # If JSON fails, wrap again safely
        cleaned = {
            "user_query": "",
            "assistant_answer": ""
        }

    return cleaned
