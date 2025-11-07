# RAG Chatbot Service

A simple AI assistant chatbot built with FastAPI that uses Retrieval-Augmented Generation (RAG).
It loads documents from the `data/` folder, builds embeddings with OpenAI, retrieves the most
relevant chunks for a query, and then asks an LLM to answer using that context.

This is perfect as a portfolio project to demonstrate:
- FastAPI backend skills
- Using OpenAI APIs
- Basic RAG design (embed -> retrieve -> generate)
- Clean, production-style structure

## Features

- REST API with FastAPI
- `/health` endpoint
- `/chat` endpoint that:
  - embeds the user question
  - retrieves top-matching document chunks
  - calls an OpenAI chat model
  - returns answer + sources
- Pluggable RAG pipeline (`app/rag.py`)
- Example documents in `data/` (resume + job description)

## Setup

1. Clone this repo and move into the folder:

```bash
git clone <your-repo-url>.git
cd rag-chatbot-service
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
```

You can change the model names to whatever you have access to.

## Running the API

```bash
uvicorn app.main:app --reload
```

The API will be available at: http://localhost:8000

- Health check: `GET /health`
- Chat: `POST /chat`

## Chat API

Request:

```http
POST /chat
Content-Type: application/json

{
  "message": "What kind of experience does this candidate have with Java?",
  "history": []
}
```

Response (example):

```json
{
  "answer": "The candidate has 4+ years of experience with Java and Spring Boot working on backend services...",
  "sources": [
    {
      "id": "sample_resume.txt#0",
      "score": 0.87,
      "metadata": {
        "filename": "sample_resume.txt"
      }
    }
  ]
}
```

## How RAG Works Here

1. On startup, the app loads all `.txt` and `.md` files from `data/`.
2. Each document is split into overlapping chunks.
3. Each chunk is embedded with an OpenAI embedding model and kept in memory.
4. For each user question:
   - The question is embedded.
   - Cosine similarity is computed against all chunks.
   - Top `k` chunks are selected as context.
   - The LLM is prompted with the context + question.
5. The answer and the sources used are returned.

You can easily extend this to load your own documents (e.g., PDFs you convert to text,
product docs, knowledge base, etc.).
