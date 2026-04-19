"""
Production RAG Pipeline.

Advanced RAG implementation with optimizations:
- Semantic chunking
- Embedding caching
- Reranking
- MMR for diversity
- Query enhancement
"""

from dataclasses import dataclass
from typing import List

from rag.config.production_config import ProductionRAGConfig
from rag.core.document_processor import create_processor
from rag.core.embeddings import create_embedding_model
from rag.core.retriever import create_retriever
from rag.core.vector_store import SearchResult, create_vector_store


@dataclass
class ProductionRAGResponse:
    """Enhanced response from production RAG pipeline."""

    query: str
    enhanced_queries: List[str]
    retrieved_chunks: List[SearchResult]
    reranked_chunks: List[SearchResult]
    answer: str
    metadata: dict


class ProductionRAGPipeline:
    """
    Production-grade RAG implementation.

    Focus: Quality, performance, and production readiness.
    """

    def __init__(self, config: ProductionRAGConfig = None):
        self.config = config or ProductionRAGConfig()

        # Initialize components
        self.document_processor = create_processor(
            strategy=self.config.chunking_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
            max_chunk_size=self.config.max_chunk_size,
        )

        self.embedding_model = create_embedding_model(
            model_type="sentence-transformers",
            model_name=self.config.embedding_model,
            use_cache=self.config.use_embedding_cache,
        )

        self.vector_store = create_vector_store(
            store_type=self.config.vector_store_type,
            persist_directory=self.config.persist_directory,
            collection_name=self.config.collection_name,
        )

        # Initialize retriever based on config
        if self.config.use_reranking:
            self.retriever = create_retriever(
                retriever_type="reranked",
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                reranker_model=self.config.reranker_model,
                initial_k=self.config.top_k,
            )
        elif self.config.use_mmr:
            self.retriever = create_retriever(
                retriever_type="mmr",
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                diversity_score=self.config.mmr_diversity_score,
                initial_k=self.config.top_k,
            )
        elif self.config.use_hybrid_search:
            self.retriever = create_retriever(
                retriever_type="hybrid",
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                alpha=self.config.hybrid_alpha,
            )
        else:
            self.retriever = create_retriever(
                retriever_type="simple",
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
            )

        self.processed_document = None

    def ingest_pdf(self, pdf_path: str) -> dict:
        """
        Ingest a PDF document with advanced processing.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with ingestion statistics
        """
        # Process document
        self.processed_document = self.document_processor.process_pdf(pdf_path)

        # Extract texts and metadata
        texts = [chunk.text for chunk in self.processed_document.chunks]
        metadatas = [chunk.metadata for chunk in self.processed_document.chunks]

        # Enrich metadata if configured
        if self.config.extract_metadata:
            metadatas = self._enrich_metadata(metadatas)

        # Generate embeddings (with caching if enabled)
        embeddings = self.embedding_model.embed_texts(texts)

        # Store in vector database
        self.vector_store.add_texts(texts, embeddings, metadatas)

        return {
            "num_chunks": self.processed_document.total_chunks,
            "num_pages": self.processed_document.metadata.get("num_pages", 0),
            "embedding_dim": self.embedding_model.dimension,
            "chunking_strategy": self.config.chunking_strategy,
            "avg_chunk_size": sum(len(t) for t in texts) / len(texts) if texts else 0,
        }

    def query(self, query_text: str, k: int = None) -> ProductionRAGResponse:
        """
        Query the RAG system with advanced features.

        Args:
            query_text: User's question
            k: Number of final chunks to return (default: config.final_k)

        Returns:
            ProductionRAGResponse with enhanced retrieval and generation
        """
        if k is None:
            k = self.config.final_k

        # Query enhancement
        enhanced_queries = self._enhance_query(query_text)

        # Retrieve chunks (initial retrieval)
        all_chunks = []
        for query in enhanced_queries:
            chunks = self.retriever.retrieve(query, k=self.config.top_k)
            all_chunks.extend(chunks)

        # Deduplicate and merge
        unique_chunks = self._deduplicate_chunks(all_chunks)

        # Rerank if enabled (and not already using reranking retriever)
        if self.config.use_reranking and not isinstance(self.retriever.__class__.__name__, "RerankedRetriever"):
            reranked_chunks = self._rerank_chunks(query_text, unique_chunks)
        else:
            reranked_chunks = unique_chunks

        # Apply MMR for diversity if enabled
        if self.config.use_mmr:
            final_chunks = self._apply_mmr(query_text, reranked_chunks, k)
        else:
            final_chunks = reranked_chunks[:k]

        # Generate answer
        answer = self._generate_answer(query_text, final_chunks)

        return ProductionRAGResponse(
            query=query_text,
            enhanced_queries=enhanced_queries,
            retrieved_chunks=unique_chunks[:self.config.top_k],
            reranked_chunks=final_chunks,
            answer=answer,
            metadata={
                "num_enhanced_queries": len(enhanced_queries),
                "num_initial_retrieved": len(unique_chunks),
                "num_final_chunks": len(final_chunks),
                "retrieval_strategy": self.config.retrieval_strategy,
            },
        )

    def _enhance_query(self, query: str) -> List[str]:
        """
        Enhance query using various techniques.

        - Multi-query: Generate multiple variations
        - HyDE: Generate hypothetical documents (if enabled)
        """
        queries = [query]  # Original query

        if not self.config.use_query_rewriting and not self.config.use_multi_query:
            return queries

        if self.config.use_multi_query:
            # Simple query variations (can be enhanced with LLM)
            queries.extend(self._generate_query_variations(query))

        return queries

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations (simplified version)."""
        # In production, this would use an LLM
        # For now, simple variations
        variations = []

        # Add question variations
        if not query.endswith("?"):
            variations.append(f"{query}?")

        # Add "what is" prefix if not present
        if not query.lower().startswith(("what", "how", "why", "when", "where", "who")):
            variations.append(f"What is {query}")

        return variations[:self.config.num_generated_queries - 1]

    def _deduplicate_chunks(self, chunks: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate chunks based on chunk_id."""
        seen_ids = set()
        unique_chunks = []

        for chunk in chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                unique_chunks.append(chunk)

        # Sort by score
        unique_chunks.sort(key=lambda x: x.score, reverse=True)
        return unique_chunks

    def _rerank_chunks(
        self,
        query: str,
        chunks: List[SearchResult],
    ) -> List[SearchResult]:
        """Rerank chunks using cross-encoder."""
        if not chunks:
            return []

        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(self.config.reranker_model)

            pairs = [(query, chunk.text) for chunk in chunks]
            rerank_scores = reranker.predict(pairs)

            # Update scores
            for chunk, score in zip(chunks, rerank_scores):
                chunk.score = float(score)

            # Resort
            chunks.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            print(f"Reranking failed: {e}")

        return chunks

    def _apply_mmr(
        self,
        query: str,
        chunks: List[SearchResult],
        k: int,
    ) -> List[SearchResult]:
        """Apply MMR for diversity (simplified)."""
        # This is a placeholder - full implementation would use embeddings
        # For now, just return top k
        return chunks[:k]

    def _enrich_metadata(self, metadatas: List[dict]) -> List[dict]:
        """Enrich chunk metadata with additional information."""
        # Can add page numbers, section headers, etc.
        return metadatas

    def _generate_answer(
        self,
        query: str,
        chunks: List[SearchResult],
    ) -> str:
        """
        Generate answer using LLM with optimized prompt.

        Production-grade prompt engineering.
        """
        if not chunks:
            return "No relevant information found in the document."

        # Build enhanced context
        context = "\n\n".join([
            f"[Source {i+1}] (Relevance: {chunk.score:.2f})\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        ])

        # Production prompt
        prompt = f"""You are an expert assistant. Answer the question based solely on the provided context.

Guidelines:
- Be precise and factual
- Cite sources using [Source N] notation
- If the context doesn't contain enough information, say so
- Be concise but complete

Context:
{context}

Question: {query}

Answer:"""

        if self.config.llm_provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI()
                response = client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": "You are an expert assistant that provides accurate, well-sourced answers based on given context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error generating answer: {str(e)}\n\nRetrieved context:\n{context}"
        else:
            # Fallback
            return f"Retrieved relevant information:\n\n{context}\n\n(Note: LLM generation not configured.)"

    def clear(self):
        """Clear the vector store and reset pipeline."""
        self.vector_store.clear()
        self.processed_document = None

    def get_embeddings_for_visualization(self):
        """Get all embeddings for visualization."""
        return self.vector_store.get_all_embeddings()
