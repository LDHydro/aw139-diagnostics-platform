"""Local inference: vLLM chat, embeddings and reranking on the RTX 5090."""

from .client import ChatMessage, Completion, InferenceError, LlmClient
from .embeddings import (
    EmbeddingClient,
    RerankClient,
    RerankResult,
    get_embedding_client,
    get_rerank_client,
)
from .router import GenerationProfile, ModelRouter, TaskKind, get_router

__all__ = [
    "ChatMessage",
    "Completion",
    "EmbeddingClient",
    "GenerationProfile",
    "InferenceError",
    "LlmClient",
    "ModelRouter",
    "RerankClient",
    "RerankResult",
    "TaskKind",
    "get_embedding_client",
    "get_rerank_client",
    "get_router",
]
