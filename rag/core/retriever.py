"""
Retriever module implementing various retrieval strategies.

Supports:
- Simple retrieval (naive RAG)
- Reranking (production RAG)
- MMR (Maximal Marginal Relevance) for diversity
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from rag.core.embeddings import EmbeddingModel
from rag.core.vector_store import SearchResult, VectorStore


class Retriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[SearchResult]:
        """Retrieve relevant documents for a query."""
        pass


class SimpleRetriever(Retriever):
    """
    Naive retrieval: direct vector similarity search.

    No reranking, no query enhancement.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, k: int = 5) -> List[SearchResult]:
        """Retrieve top-k documents by cosine similarity."""
        # Embed query
        query_embedding = self.embedding_model.embed_query(query)

        # Search vector store
        results = self.vector_store.similarity_search(query_embedding, k=k)

        return results


class RerankedRetriever(Retriever):
    """
    Production retrieval with reranking.

    Two-stage retrieval:
    1. Initial retrieval of top_k candidates
    2. Rerank using cross-encoder for higher precision
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_k: int = 10,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.initial_k = initial_k

        # Load reranker (cross-encoder)
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    def retrieve(self, query: str, k: int = 5) -> List[SearchResult]:
        """Retrieve and rerank documents."""
        # Stage 1: Initial retrieval
        query_embedding = self.embedding_model.embed_query(query)
        initial_results = self.vector_store.similarity_search(
            query_embedding,
            k=self.initial_k,
        )

        if not initial_results:
            return []

        # Stage 2: Rerank with cross-encoder
        pairs = [(query, result.text) for result in initial_results]
        rerank_scores = self.reranker.predict(pairs)

        # Update scores and resort
        for result, score in zip(initial_results, rerank_scores):
            result.score = float(score)

        reranked_results = sorted(
            initial_results,
            key=lambda x: x.score,
            reverse=True,
        )

        return reranked_results[:k]


class MMRRetriever(Retriever):
    """
    Maximal Marginal Relevance (MMR) retriever.

    Balances relevance and diversity to avoid redundant results.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        diversity_score: float = 0.3,
        initial_k: int = 20,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.diversity_score = diversity_score  # lambda in MMR formula
        self.initial_k = initial_k

    def retrieve(self, query: str, k: int = 5) -> List[SearchResult]:
        """Retrieve documents using MMR for diversity."""
        # Initial retrieval
        query_embedding = self.embedding_model.embed_query(query)
        initial_results = self.vector_store.similarity_search(
            query_embedding,
            k=self.initial_k,
        )

        if not initial_results:
            return []

        # Extract embeddings for initial results
        all_embeddings, all_texts, _ = self.vector_store.get_all_embeddings()
        result_indices = [r.chunk_id for r in initial_results]
        result_embeddings = all_embeddings[result_indices]

        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(initial_results)))

        for _ in range(min(k, len(initial_results))):
            if not remaining_indices:
                break

            # Calculate MMR scores
            mmr_scores = []
            for idx in remaining_indices:
                # Relevance to query
                relevance = self._cosine_similarity(
                    query_embedding,
                    result_embeddings[idx],
                )

                # Diversity (max similarity to already selected)
                if selected_indices:
                    selected_embeddings = result_embeddings[selected_indices]
                    max_similarity = max(
                        self._cosine_similarity(
                            result_embeddings[idx],
                            selected_embeddings[i],
                        )
                        for i in range(len(selected_indices))
                    )
                else:
                    max_similarity = 0

                # MMR score: λ * relevance - (1-λ) * diversity
                mmr_score = (
                    self.diversity_score * relevance
                    - (1 - self.diversity_score) * max_similarity
                )
                mmr_scores.append(mmr_score)

            # Select best MMR score
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Return selected results
        return [initial_results[i] for i in selected_indices]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(
            np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        )


class HybridRetriever(Retriever):
    """
    Hybrid retrieval combining vector search with keyword search.

    Production-grade: blends semantic and lexical matching.
    Note: Simplified implementation. Full hybrid search would use BM25.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        alpha: float = 0.5,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.alpha = alpha  # 0=keyword only, 1=semantic only

    def retrieve(self, query: str, k: int = 5) -> List[SearchResult]:
        """
        Hybrid retrieval with semantic and keyword matching.

        Note: This is a simplified version. Production would use BM25 for keyword search.
        """
        # Semantic search
        query_embedding = self.embedding_model.embed_query(query)
        semantic_results = self.vector_store.similarity_search(query_embedding, k=k*2)

        # Simple keyword matching (count query term overlap)
        query_terms = set(query.lower().split())
        for result in semantic_results:
            doc_terms = set(result.text.lower().split())
            keyword_score = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0

            # Combine scores
            result.score = (
                self.alpha * result.score +
                (1 - self.alpha) * keyword_score
            )

        # Resort and return top k
        semantic_results.sort(key=lambda x: x.score, reverse=True)
        return semantic_results[:k]


def create_retriever(
    retriever_type: str,
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
    **kwargs,
) -> Retriever:
    """
    Factory function to create a retriever.

    Args:
        retriever_type: 'simple', 'reranked', 'mmr', or 'hybrid'
        vector_store: Vector store instance
        embedding_model: Embedding model instance
        **kwargs: Additional parameters for the retriever

    Returns:
        Configured Retriever instance
    """
    if retriever_type == "simple":
        return SimpleRetriever(vector_store, embedding_model)

    elif retriever_type == "reranked":
        return RerankedRetriever(
            vector_store,
            embedding_model,
            reranker_model_name=kwargs.get(
                "reranker_model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            initial_k=kwargs.get("initial_k", 10),
        )

    elif retriever_type == "mmr":
        return MMRRetriever(
            vector_store,
            embedding_model,
            diversity_score=kwargs.get("diversity_score", 0.3),
            initial_k=kwargs.get("initial_k", 20),
        )

    elif retriever_type == "hybrid":
        return HybridRetriever(
            vector_store,
            embedding_model,
            alpha=kwargs.get("alpha", 0.5),
        )

    else:
        raise ValueError(f"Unknown retriever_type: {retriever_type}")
