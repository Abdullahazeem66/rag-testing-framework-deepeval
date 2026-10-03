"""HTTP API.  Run:  uvicorn rag_app.api:app --reload"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from rag_app.assistant import RAGAssistant
from rag_app.ingest import build_index

app = FastAPI(title="Orbitly RAG Assistant")
build_index()
assistant = RAGAssistant()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


class ChatReply(BaseModel):
    answer: str
    standalone_query: str
    sources: list[str]
    retrieval_context: list[str]
    latency_s: float


@app.post("/chat", response_model=ChatReply)
def chat(req: ChatRequest) -> ChatReply:
    resp = assistant.chat(req.message, [m.model_dump() for m in req.history])
    return ChatReply(
        answer=resp.answer,
        standalone_query=resp.standalone_query,
        sources=[c.chunk_id for c in resp.chunks],
        retrieval_context=resp.retrieval_context,
        latency_s=resp.latency_s,
    )
