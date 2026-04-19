"""
Configuration for Production RAG Pipeline.

Advanced settings with optimizations for accuracy, performance, and cost.
Focus: Production-grade quality, sophisticated retrieval strategies.
"""

from pydantic import BaseModel, Field
from typing import Literal


class ProductionRAGConfig(BaseModel):
    """Configuration for Production RAG implementation."""

    # Document Processing
    chunk_size: int = Field(
        default=1000,
        description="Larger semantic chunks for better context"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap to preserve context across chunks"
    )
    chunking_strategy: Literal["semantic", "recursive"] = "semantic"
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

    # Metadata Enrichment
    extract_metadata: bool = True
    include_page_numbers: bool = True
    include_section_headers: bool = True

    # Embeddings
    embedding_model: str = Field(
        default="all-mpnet-base-v2",
        description="Higher quality sentence-transformers model"
    )
    embedding_dimension: int = 768
    normalize_embeddings: bool = True
    batch_size: int = 32

    # Vector Store
    vector_store_type: Literal["chromadb", "qdrant"] = "chromadb"
    persist_directory: str = "./rag/data/vectorstore"
    collection_name: str = "production_rag"

    # Retrieval
    retrieval_strategy: Literal["hybrid"] = "hybrid"
    top_k: int = Field(default=10, description="Initial retrieval count (pre-reranking)")
    final_k: int = Field(default=5, description="Final chunks after reranking")
    similarity_metric: Literal["cosine"] = "cosine"

    # Advanced Retrieval Features
    use_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_mmr: bool = True
    mmr_diversity_score: float = 0.3
    use_hybrid_search: bool = True
    hybrid_alpha: float = 0.5  # 0=keyword only, 1=semantic only

    # Query Enhancement
    use_query_rewriting: bool = True
    use_hyde: bool = False  # Hypothetical Document Embeddings
    use_multi_query: bool = True
    num_generated_queries: int = 3

    # LLM
    llm_provider: Literal["openai", "ollama"] = "openai"
    llm_model: str = "gpt-4-turbo-preview"
    temperature: float = 0.3
    max_tokens: int = 1000

    # Caching
    use_embedding_cache: bool = True
    use_query_cache: bool = True
    cache_ttl: int = 3600  # seconds

    # Performance
    enable_async: bool = True
    max_concurrent_embeddings: int = 5

    class Config:
        frozen = True


# Default instance
PRODUCTION_CONFIG = ProductionRAGConfig()
