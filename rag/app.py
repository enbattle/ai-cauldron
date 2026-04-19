"""
Streamlit application for Visual RAG Pipeline.

Allows users to:
- Upload PDFs
- Choose between Naive and Production RAG
- Visualize embeddings, retrieval, and search process
- Query documents and see results
"""

import sys
from pathlib import Path

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config.naive_config import NAIVE_CONFIG
from rag.config.production_config import PRODUCTION_CONFIG
from rag.pipelines.naive_rag import NaiveRAGPipeline
from rag.pipelines.production_rag import ProductionRAGPipeline
from rag.utils.helpers import ensure_directory, format_metadata, get_file_size
from rag.visualizations.embeddings_viz import (
    create_embedding_plot_2d,
    create_embedding_plot_3d,
)
from rag.visualizations.similarity_viz import (
    create_comparison_chart,
    create_similarity_bar_chart,
)

# Page configuration
st.set_page_config(
    page_title="Visual RAG Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure data directories exist
ensure_directory("./rag/data/uploads")
ensure_directory("./rag/data/vectorstore")


def init_session_state():
    """Initialize session state variables."""
    if "rag_mode" not in st.session_state:
        st.session_state.rag_mode = "Naive RAG"

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    if "document_ingested" not in st.session_state:
        st.session_state.document_ingested = False

    if "ingestion_stats" not in st.session_state:
        st.session_state.ingestion_stats = None

    if "query_results" not in st.session_state:
        st.session_state.query_results = None


def create_pipeline(mode: str):
    """Create a RAG pipeline based on selected mode."""
    if mode == "Naive RAG":
        return NaiveRAGPipeline(NAIVE_CONFIG)
    else:
        return ProductionRAGPipeline(PRODUCTION_CONFIG)


def display_mode_comparison():
    """Display comparison between Naive and Production RAG."""
    st.subheader("📊 Naive vs Production RAG Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 Naive RAG")
        st.markdown("""
        **Focus**: Simplicity and speed

        **Features**:
        - ✅ Fixed-size chunking (500 chars)
        - ✅ Basic sentence-transformers model
        - ✅ In-memory vector store
        - ✅ Simple cosine similarity
        - ❌ No reranking
        - ❌ No caching
        - ❌ No query enhancement

        **Best for**: Learning, prototyping, small documents
        """)

    with col2:
        st.markdown("### 🟢 Production RAG")
        st.markdown("""
        **Focus**: Quality and accuracy

        **Features**:
        - ✅ Semantic chunking (1000 chars + overlap)
        - ✅ Advanced embedding model
        - ✅ Persistent vector store (ChromaDB)
        - ✅ Hybrid search (semantic + keyword)
        - ✅ Cross-encoder reranking
        - ✅ MMR for diversity
        - ✅ Query enhancement
        - ✅ Embedding & query caching

        **Best for**: Production use, complex documents, high accuracy needs
        """)


def main():
    """Main application function."""
    init_session_state()

    # Header
    st.title("📚 Visual RAG Pipeline")
    st.markdown("Upload a PDF and explore the RAG (Retrieval Augmented Generation) process visually!")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Mode selection
        mode = st.radio(
            "Select RAG Mode",
            ["Naive RAG", "Production RAG"],
            index=0 if st.session_state.rag_mode == "Naive RAG" else 1,
            help="Choose between simple (Naive) or advanced (Production) RAG implementation"
        )

        # Update mode if changed
        if mode != st.session_state.rag_mode:
            st.session_state.rag_mode = mode
            st.session_state.pipeline = None
            st.session_state.document_ingested = False
            st.session_state.query_results = None

        st.markdown("---")

        # File upload
        st.subheader("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to analyze"
        )

        if uploaded_file:
            # Display file info
            file_size = len(uploaded_file.getvalue())
            st.info(f"📁 **{uploaded_file.name}**\n\n Size: {file_size / 1024:.1f} KB")

            # Ingest button
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner(f"Processing with {mode}..."):
                    # Save uploaded file
                    upload_path = Path("./rag/data/uploads") / uploaded_file.name
                    with open(upload_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Create pipeline
                    st.session_state.pipeline = create_pipeline(mode)

                    # Ingest document
                    try:
                        stats = st.session_state.pipeline.ingest_pdf(str(upload_path))
                        st.session_state.ingestion_stats = stats
                        st.session_state.document_ingested = True
                        st.success("✅ Document processed successfully!")
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        st.session_state.document_ingested = False

        st.markdown("---")

        # Settings info
        with st.expander("ℹ️ Current Settings"):
            if mode == "Naive RAG":
                st.write(f"**Chunk Size**: {NAIVE_CONFIG.chunk_size}")
                st.write(f"**Model**: {NAIVE_CONFIG.embedding_model}")
                st.write(f"**Top K**: {NAIVE_CONFIG.top_k}")
            else:
                st.write(f"**Chunk Size**: {PRODUCTION_CONFIG.chunk_size}")
                st.write(f"**Model**: {PRODUCTION_CONFIG.embedding_model}")
                st.write(f"**Retrieval**: {PRODUCTION_CONFIG.retrieval_strategy}")
                st.write(f"**Reranking**: {PRODUCTION_CONFIG.use_reranking}")

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Overview",
        "📊 Document Analysis",
        "🔍 Query & Retrieve",
        "📈 Visualizations"
    ])

    # Tab 1: Overview
    with tab1:
        display_mode_comparison()

        if st.session_state.document_ingested:
            st.success("✅ Document is loaded and ready for querying!")
            st.json(st.session_state.ingestion_stats)

    # Tab 2: Document Analysis
    with tab2:
        if not st.session_state.document_ingested:
            st.info("👈 Upload and process a document first")
        else:
            st.subheader("📊 Document Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Chunks",
                    st.session_state.ingestion_stats.get("num_chunks", 0)
                )

            with col2:
                st.metric(
                    "Pages",
                    st.session_state.ingestion_stats.get("num_pages", 0)
                )

            with col3:
                st.metric(
                    "Embedding Dim",
                    st.session_state.ingestion_stats.get("embedding_dim", 0)
                )

            # Show chunk samples
            st.subheader("📝 Sample Chunks")
            if st.session_state.pipeline.processed_document:
                chunks = st.session_state.pipeline.processed_document.chunks[:5]
                for i, chunk in enumerate(chunks):
                    with st.expander(f"Chunk {i} ({len(chunk.text)} chars)"):
                        st.write(chunk.text)

    # Tab 3: Query & Retrieve
    with tab3:
        if not st.session_state.document_ingested:
            st.info("👈 Upload and process a document first")
        else:
            st.subheader("🔍 Query Your Document")

            # Query input
            query = st.text_input(
                "Enter your question:",
                placeholder="What is this document about?",
                help="Ask a question about your uploaded document"
            )

            k = st.slider(
                "Number of chunks to retrieve",
                min_value=1,
                max_value=10,
                value=5,
                help="How many relevant chunks to retrieve"
            )

            if st.button("🔎 Search", type="primary") and query:
                with st.spinner("Retrieving relevant information..."):
                    try:
                        results = st.session_state.pipeline.query(query, k=k)
                        st.session_state.query_results = results

                        # Display answer
                        st.subheader("💡 Answer")
                        st.markdown(results.answer)

                        # Display retrieved chunks
                        st.subheader("📄 Retrieved Chunks")
                        for i, chunk in enumerate(results.retrieved_chunks):
                            with st.expander(
                                f"Chunk {chunk.chunk_id} (Score: {chunk.score:.3f})"
                            ):
                                st.write(chunk.text)
                                if chunk.metadata:
                                    st.caption(format_metadata(chunk.metadata))

                    except Exception as e:
                        st.error(f"❌ Error during retrieval: {str(e)}")

            # Show previous results if available
            elif st.session_state.query_results:
                st.subheader("💡 Previous Answer")
                st.markdown(st.session_state.query_results.answer)

    # Tab 4: Visualizations
    with tab4:
        if not st.session_state.document_ingested:
            st.info("👈 Upload and process a document first")
        else:
            st.subheader("📈 Embedding Visualizations")

            # Visualization settings
            viz_type = st.radio(
                "Visualization Type",
                ["2D", "3D"],
                horizontal=True
            )

            reduction_method = st.selectbox(
                "Dimensionality Reduction Method",
                ["umap", "tsne", "pca"],
                help="Method to reduce embeddings to 2D/3D"
            )

            # Get embeddings
            embeddings, texts, metadatas = st.session_state.pipeline.get_embeddings_for_visualization()

            # Add query embedding if available
            query_embedding = None
            query_text = None
            if st.session_state.query_results:
                query_text = st.session_state.query_results.query
                query_embedding = st.session_state.pipeline.embedding_model.embed_query(query_text)

            # Create visualization
            with st.spinner(f"Creating {viz_type} visualization..."):
                try:
                    if viz_type == "2D":
                        fig = create_embedding_plot_2d(
                            embeddings,
                            texts,
                            metadatas,
                            query_embedding=query_embedding,
                            query_text=query_text,
                            method=reduction_method,
                        )
                    else:
                        fig = create_embedding_plot_3d(
                            embeddings,
                            texts,
                            metadatas,
                            query_embedding=query_embedding,
                            query_text=query_text,
                            method=reduction_method,
                        )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")

            # Similarity scores visualization
            if st.session_state.query_results:
                st.subheader("📊 Similarity Scores")
                try:
                    fig = create_similarity_bar_chart(
                        st.session_state.query_results.retrieved_chunks
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating similarity chart: {str(e)}")


if __name__ == "__main__":
    main()
