"""
Vector store module for storing and retrieving document embeddings.

Supports multiple vector database backends:
- In-memory (simple, no persistence)
- ChromaDB (persistent, local-first)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SearchResult:
    """Result from a vector similarity search."""

    chunk_id: int
    text: str
    score: float
    metadata: dict


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        """Add texts and their embeddings to the store."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[SearchResult]:
        """Search for the k most similar documents."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all documents from the store."""
        pass

    @abstractmethod
    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str], List[dict]]:
        """Get all embeddings, texts, and metadata (for visualization)."""
        pass


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store using numpy for similarity computation.

    Fast, but no persistence. Good for naive RAG implementation.
    """

    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []
        self.metadatas: List[dict] = []
        self.ids: List[str] = []

    def add_texts(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        """Add texts and embeddings to the in-memory store."""
        if metadatas is None:
            metadatas = [{} for _ in texts]

        ids = []
        for i, (text, embedding, metadata) in enumerate(
            zip(texts, embeddings, metadatas)
        ):
            doc_id = f"doc_{len(self.ids) + i}"
            self.ids.append(doc_id)
            self.texts.append(text)
            self.embeddings.append(embedding)
            self.metadatas.append(metadata)
            ids.append(doc_id)

        return ids

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[SearchResult]:
        """Perform cosine similarity search."""
        if not self.embeddings:
            return []

        # Stack embeddings into matrix
        embedding_matrix = np.vstack(self.embeddings)

        # Compute cosine similarity
        similarities = self._cosine_similarity(query_embedding, embedding_matrix)

        # Get top k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Build results
        results = []
        for idx in top_k_indices:
            results.append(
                SearchResult(
                    chunk_id=idx,
                    text=self.texts[idx],
                    score=float(similarities[idx]),
                    metadata=self.metadatas[idx],
                )
            )

        return results

    @staticmethod
    def _cosine_similarity(query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and embeddings."""
        # Normalize
        query_norm = query / np.linalg.norm(query)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Dot product
        similarities = np.dot(embeddings_norm, query_norm)
        return similarities

    def clear(self):
        """Clear all stored data."""
        self.embeddings = []
        self.texts = []
        self.metadatas = []
        self.ids = []

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str], List[dict]]:
        """Return all embeddings, texts, and metadata."""
        if not self.embeddings:
            return np.array([]), [], []
        return np.vstack(self.embeddings), self.texts, self.metadatas


class ChromaDBVectorStore(VectorStore):
    """
    ChromaDB-based vector store with persistence.

    Better for production: persistent, scalable, supports metadata filtering.
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./rag/data/vectorstore",
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Run: pip install chromadb"
            )

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_texts(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[dict]] = None,
    ) -> List[str]:
        """Add texts and embeddings to ChromaDB."""
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Generate IDs
        current_count = self.collection.count()
        ids = [f"doc_{current_count + i}" for i in range(len(texts))]

        # Convert numpy to list for ChromaDB
        embeddings_list = embeddings.tolist()

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
        )

        return ids

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> List[SearchResult]:
        """Perform similarity search using ChromaDB."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, self.collection.count()),
        )

        # Parse results
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Extract chunk_id from doc_id (format: "doc_X")
                chunk_id = int(doc_id.split("_")[1])

                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        text=results["documents"][0][i],
                        score=1.0 - results["distances"][0][i],  # Convert distance to similarity
                        metadata=results["metadatas"][0][i] if results["metadatas"][0] else {},
                    )
                )

        return search_results

    def clear(self):
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str], List[dict]]:
        """Get all embeddings, texts, and metadata."""
        count = self.collection.count()
        if count == 0:
            return np.array([]), [], []

        # Retrieve all documents
        results = self.collection.get(
            include=["embeddings", "documents", "metadatas"]
        )

        embeddings = np.array(results["embeddings"])
        texts = results["documents"]
        metadatas = results["metadatas"]

        return embeddings, texts, metadatas


def create_vector_store(
    store_type: str = "in-memory",
    collection_name: str = "rag_documents",
    persist_directory: str = "./rag/data/vectorstore",
) -> VectorStore:
    """
    Factory function to create a vector store.

    Args:
        store_type: 'in-memory' or 'chromadb'
        collection_name: Name for the collection (ChromaDB only)
        persist_directory: Directory for persistence (ChromaDB only)

    Returns:
        Configured VectorStore instance
    """
    if store_type == "in-memory":
        return InMemoryVectorStore()
    elif store_type == "chromadb":
        return ChromaDBVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    else:
        raise ValueError(f"Unknown store_type: {store_type}")
