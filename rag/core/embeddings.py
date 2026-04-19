"""
Embeddings module for converting text to vector representations.

Supports multiple embedding providers:
- sentence-transformers (local, free)
- OpenAI (API-based, high quality)
"""

import hashlib
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts into vectors."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of embeddings."""
        pass


class SentenceTransformerEmbeddings(EmbeddingModel):
    """
    Embeddings using sentence-transformers library.

    Popular models:
    - all-MiniLM-L6-v2: Fast, lightweight (384 dim)
    - all-mpnet-base-v2: High quality (768 dim)
    - all-MiniLM-L12-v2: Balanced (384 dim)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.array(embedding)

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbeddings(EmbeddingModel):
    """
    Embeddings using OpenAI API.

    Models:
    - text-embedding-3-small: 1536 dim, cost-effective
    - text-embedding-3-large: 3072 dim, highest quality
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Run: pip install openai"
            )

        self.model = model
        self.client = OpenAI(api_key=api_key)

        # Determine dimensions based on model
        if "small" in model:
            self._dimension = 1536
        elif "large" in model:
            self._dimension = 3072
        else:
            self._dimension = 1536  # Default

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts using OpenAI API."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        response = self.client.embeddings.create(
            input=[query],
            model=self.model,
        )
        return np.array(response.data[0].embedding)

    @property
    def dimension(self) -> int:
        return self._dimension


class CachedEmbeddingModel:
    """
    Wrapper that adds caching to any embedding model.

    Useful for production scenarios where the same texts may be embedded multiple times.
    """

    def __init__(
        self,
        base_model: EmbeddingModel,
        cache_dir: str = "./rag/data/embedding_cache",
    ):
        self.base_model = base_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}  # In-memory cache
        self._load_cache()

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "embeddings.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._cache = pickle.load(f)
            except Exception as e:
                print(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "embeddings.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts with caching."""
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        # Check cache
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                embeddings.append(self._cache[cache_key])
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                embeddings.append(None)  # Placeholder

        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self.base_model.embed_texts(texts_to_embed)

            # Update cache and results
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                cache_key = self._get_cache_key(texts[idx])
                self._cache[cache_key] = embedding
                embeddings[idx] = embedding

            self._save_cache()

        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query with caching."""
        cache_key = self._get_cache_key(query)

        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self.base_model.embed_query(query)
        self._cache[cache_key] = embedding
        self._save_cache()

        return embedding

    @property
    def dimension(self) -> int:
        return self.base_model.dimension


def create_embedding_model(
    model_type: str = "sentence-transformers",
    model_name: str = "all-MiniLM-L6-v2",
    use_cache: bool = False,
    **kwargs,
) -> EmbeddingModel:
    """
    Factory function to create an embedding model.

    Args:
        model_type: 'sentence-transformers' or 'openai'
        model_name: Specific model name
        use_cache: Whether to enable caching
        **kwargs: Additional parameters for the model

    Returns:
        Configured EmbeddingModel instance
    """
    if model_type == "sentence-transformers":
        base_model = SentenceTransformerEmbeddings(
            model_name=model_name,
            normalize=kwargs.get("normalize", True),
            device=kwargs.get("device", None),
        )
    elif model_type == "openai":
        base_model = OpenAIEmbeddings(
            model=model_name,
            api_key=kwargs.get("api_key", None),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if use_cache:
        return CachedEmbeddingModel(
            base_model=base_model,
            cache_dir=kwargs.get("cache_dir", "./rag/data/embedding_cache"),
        )

    return base_model
