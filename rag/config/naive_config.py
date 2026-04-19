"""
Configuration for Naive RAG Pipeline.

Simple, straightforward settings with minimal optimization.
Focus: Quick implementation, easy to understand.
"""

from pydantic import BaseModel, Field
from typing import Literal


class NaiveRAGConfig(BaseModel):
    """Configuration for Naive RAG implementation."""

    # Document Processing
    chunk_size: int = Field(default=500, description="Fixed character chunk size")
    chunk_overlap: int = Field(default=0, description="No overlap between chunks")
    chunking_strategy: Literal["fixed"] = "fixed"

    # Embeddings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Lightweight sentence-transformers model"
    )
    embedding_dimension: int = 384
    normalize_embeddings: bool = True

    # Vector Store
    vector_store_type: Literal["in-memory"] = "in-memory"
    persist_directory: str | None = None

    # Retrieval
    retrieval_strategy: Literal["simple"] = "simple"
    top_k: int = Field(default=3, description="Number of chunks to retrieve")
    similarity_metric: Literal["cosine"] = "cosine"
    use_reranking: bool = False
    use_mmr: bool = False

    # LLM
    llm_provider: Literal["openai", "ollama"] = "openai"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500

    # Caching
    use_embedding_cache: bool = False
    use_query_cache: bool = False

    class Config:
        frozen = True


# Default instance
NAIVE_CONFIG = NaiveRAGConfig()
