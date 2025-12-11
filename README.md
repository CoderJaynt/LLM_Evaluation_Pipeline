# LLM Evaluation Pipeline

This repository contains an evaluation system for Large Language Model (LLM) responses.
Given two files:

chat.json — a raw conversation between user and assistant (may be unstructured or invalid JSON)

context.json — retrieved context chunks from a vector database

The pipeline extracts the relevant messages, computes embeddings, and produces evaluation scores for:

->Relevance

->Completeness

->Hallucination

->Latency

A Streamlit-based UI is also included for testing and demonstration.

### 1. Project Structure
llm-eval/
│
├── pipeline.py
├── app.py
├── json_cleaner.py
├── requirements.txt
├── README.md
│
├── sample_data/
│   ├── chat-1.json
│   └── context-1.json
│
└── utils/
    ├── embeddings.py
    └── scoring.py

