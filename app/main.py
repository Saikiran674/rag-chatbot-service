from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .models import ChatRequest, ChatResponse, Source, SourceMetadata
from .rag import pipeline

app = FastAPI(title="RAG Chatbot Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Build the in-memory index when the app starts
    pipeline.load_documents()
    pipeline.build_index()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    history = [{"role": h.role, "content": h.content} for h in request.history]
    result = pipeline.generate_answer(request.message, history=history, top_k=5)

    sources_models = [
        Source(
            id=s["id"],
            score=s["score"],
            metadata=SourceMetadata(**s["metadata"])
        )
        for s in result["sources"]
    ]

    return ChatResponse(
        answer=result["answer"],
        sources=sources_models,
    )
