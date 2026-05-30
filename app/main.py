from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.rag import answer_question

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="Supabase Support Bot",
    description="RAG-powered support bot trained on the public Supabase documentation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
@app.head("/health")
def health() -> Response:
    return Response(content='{"status":"ok"}', media_type="application/json")


@app.get("/")
def root() -> FileResponse:
    widget = STATIC_DIR / "widget.html"
    if not widget.exists():
        raise HTTPException(status_code=404, detail="widget not built")
    return FileResponse(widget)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = answer_question(req.message)
    return ChatResponse(answer=result.answer, sources=result.sources)
