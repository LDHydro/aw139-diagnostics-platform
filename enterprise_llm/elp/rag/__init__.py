"""Retrieval-augmented question answering over department governing documents."""

from .answer import AnswerSynthesizer, GroundedAnswer, get_synthesizer
from .chunker import Chunk, chunk_blocks, count_tokens
from .ingest import Ingestor, IngestRequest, IngestResult, delete_document
from .parsers import Block, ParsedDocument, ParseError, parse
from .retrieve import Passage, RetrievalFilter, Retriever, get_retriever

__all__ = [
    "AnswerSynthesizer",
    "Block",
    "Chunk",
    "GroundedAnswer",
    "IngestRequest",
    "IngestResult",
    "Ingestor",
    "ParseError",
    "ParsedDocument",
    "Passage",
    "RetrievalFilter",
    "Retriever",
    "chunk_blocks",
    "count_tokens",
    "delete_document",
    "get_retriever",
    "get_synthesizer",
    "parse",
]
