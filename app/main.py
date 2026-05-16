"""FastAPI entry point. Exposes GET /health and POST /chat."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.agent import chat
from app.config import MAX_TURNS, configure_logging
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.retriever import get_retriever

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the FAISS index + embedding model so the first /chat call
    # isn't paying that cost in user-facing latency.
    logger.info("Warming up retriever...")
    try:
        get_retriever()
        logger.info("Retriever ready.")
    except FileNotFoundError as exc:
        logger.error(
            "Index not built — server will run but /chat will fail. Run: "
            "python -m scripts.build_index. (%s)",
            exc,
        )
    yield


app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL assessments from the official "
        "SHL product catalog. Backed by FAISS semantic retrieval + Gemini Flash."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    if request.messages[-1].role != "user":
        raise HTTPException(
            status_code=400, detail="The final message must have role='user'."
        )

    messages = [m.model_dump() for m in request.messages][-MAX_TURNS:]
    try:
        result = chat(messages)
    except Exception:  # pragma: no cover - last-resort handler
        logger.exception("Unhandled error in chat endpoint")
        return ChatResponse(
            reply="Something went wrong on my side. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )
    return ChatResponse(**result)
