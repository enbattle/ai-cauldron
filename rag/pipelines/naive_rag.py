"""
Naive RAG Pipeline.

Simple, straightforward RAG implementation:
- Fixed-size chunking
- Direct embedding
- Simple cosine similarity retrieval
- No caching or optimization
"""

from dataclasses import dataclass
from typing import List

from rag.config.naive_config import NaiveRAGConfig
from rag.core.document_processor import create_processor
from rag.core.embeddings import create_embedding_model
from rag.core.retriever import create_retriever
from rag.core.vector_store import SearchResult, create_vector_store


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""

    query: str
    retrieved_chunks: List[SearchResult]
    answer: str
    metadata: dict


class NaiveRAGPipeline:
    """
    Naive RAG implementation.

    Focus: Simplicity and ease of understanding.
    """

    def __init__(self, config: NaiveRAGConfig = None):
        self.config = config or NaiveRAGConfig()

        # Initialize components
        self.document_processor = create_processor(
            strategy=self.config.chunking_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        self.embedding_model = create_embedding_model(
            model_type="sentence-transformers",
            model_name=self.config.embedding_model,
            use_cache=self.config.use_embedding_cache,
        )

        self.vector_store = create_vector_store(
            store_type=self.config.vector_store_type,
        )

        self.retriever = create_retriever(
            retriever_type=self.config.retrieval_strategy,
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
        )

        self.processed_document = None

    def ingest_pdf(self, pdf_path: str) -> dict:
        """
        Ingest a PDF document into the RAG system.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with ingestion statistics
        """
        # Process document
        self.processed_document = self.document_processor.process_pdf(pdf_path)

        # Extract texts for embedding
        texts = [chunk.text for chunk in self.processed_document.chunks]
        metadatas = [chunk.metadata for chunk in self.processed_document.chunks]

        # Generate embeddings
        embeddings = self.embedding_model.embed_texts(texts)

        # Store in vector database
        self.vector_store.add_texts(texts, embeddings, metadatas)

        return {
            "num_chunks": self.processed_document.total_chunks,
            "num_pages": self.processed_document.metadata.get("num_pages", 0),
            "embedding_dim": self.embedding_model.dimension,
        }

    def query(self, query_text: str, k: int = None) -> RAGResponse:
        """
        Query the RAG system.

        Args:
            query_text: User's question
            k: Number of chunks to retrieve (default: config.top_k)

        Returns:
            RAGResponse with retrieved chunks and generated answer
        """
        if k is None:
            k = self.config.top_k

        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query_text, k=k)

        # Generate answer (simple context concatenation)
        answer = self._generate_answer(query_text, retrieved_chunks)

        return RAGResponse(
            query=query_text,
            retrieved_chunks=retrieved_chunks,
            answer=answer,
            metadata={
                "num_retrieved": len(retrieved_chunks),
                "retrieval_strategy": self.config.retrieval_strategy,
            },
        )

    def _generate_answer(
        self,
        query: str,
        chunks: List[SearchResult],
    ) -> str:
        """
        Generate answer from retrieved chunks.

        Naive implementation: Simple prompt with no optimization.
        """
        if not chunks:
            return "No relevant information found in the document."

        # Build context
        context = "\n\n".join([f"[{i+1}] {chunk.text}" for i, chunk in enumerate(chunks)])

        # Simple prompt
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

        # For naive RAG, we'll return a simple formatted response
        # In production, this would call an LLM
        if self.config.llm_provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI()
                response = client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error generating answer: {str(e)}\n\nRetrieved context:\n{context}"
        else:
            # Fallback: return context without LLM
            return f"Retrieved relevant information:\n\n{context}\n\n(Note: LLM generation not configured. Install and configure OpenAI or Ollama to get generated answers.)"

    def clear(self):
        """Clear the vector store and reset pipeline."""
        self.vector_store.clear()
        self.processed_document = None

    def get_embeddings_for_visualization(self):
        """Get all embeddings for visualization purposes."""
        return self.vector_store.get_all_embeddings()
