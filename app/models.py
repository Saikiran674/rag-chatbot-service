from pydantic import BaseModel
from typing import List, Optional, Literal

class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatHistoryMessage] = []

class SourceMetadata(BaseModel):
    filename: str

class Source(BaseModel):
    id: str
    score: float
    metadata: SourceMetadata

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
