"""
LLM-Powered Intelligent Query-Retrieval System
Source package initialization
"""

from .document_processor import DocumentProcessor
from .embedding_engine import EmbeddingEngine
from .llm_handler import LLMHandler
from .clause_matcher import ClauseMatcher
from .response_generator import ResponseGenerator

__version__ = "1.0.0"
__author__ = "Hackathon Team"

__all__ = [
    "DocumentProcessor",
    "EmbeddingEngine", 
    "LLMHandler",
    "ClauseMatcher",
    "ResponseGenerator"
]
