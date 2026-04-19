# 📚 Visual RAG Pipeline

An interactive visual demonstration of RAG (Retrieval Augmented Generation) pipelines, comparing **Naive** and **Production** implementations.

## 🎯 Overview

This project provides a hands-on learning environment to understand how RAG systems work by visualizing:
- PDF document processing and chunking
- Text embeddings in 2D/3D space
- Semantic search and similarity scoring
- Retrieval and generation process

## ✨ Features

### 🔵 Naive RAG
Simple, straightforward implementation for learning:
- Fixed-size text chunking
- Basic embedding model (all-MiniLM-L6-v2)
- In-memory vector storage
- Direct cosine similarity search
- No caching or optimization

### 🟢 Production RAG
Advanced implementation with production-ready features:
- Semantic chunking with overlap
- High-quality embeddings (all-mpnet-base-v2)
- Persistent vector store (ChromaDB)
- Hybrid search (semantic + keyword)
- Cross-encoder reranking
- MMR (Maximal Marginal Relevance) for diversity
- Query enhancement
- Embedding and query caching

### 📊 Visualizations
- **2D/3D Embedding Plots**: Visualize document chunks in reduced dimensional space using UMAP, t-SNE, or PCA
- **Similarity Scores**: Bar charts showing retrieval relevance
- **Interactive Exploration**: Click and explore chunks, see query positioning

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory**:
```bash
cd rag
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional, for LLM features):

Create a `.env` file in the `rag` directory:
```bash
# For OpenAI (optional - for answer generation)
OPENAI_API_KEY=your_api_key_here
```

### Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Choose RAG Mode
- Select **Naive RAG** for a simple, easy-to-understand implementation
- Select **Production RAG** for advanced features and higher quality

### Step 2: Upload a PDF
- Click "Browse files" or drag-and-drop a PDF document
- Click "Process Document" to ingest it into the system

### Step 3: Explore
- **Overview Tab**: See mode comparison and ingestion stats
- **Document Analysis Tab**: View chunking results and sample chunks
- **Query & Retrieve Tab**: Ask questions and see retrieved context
- **Visualizations Tab**: Explore embeddings in 2D/3D space

### Step 4: Query Your Document
1. Enter a question in the query box
2. Adjust the number of chunks to retrieve (k)
3. Click "Search" to see results
4. View the generated answer and retrieved chunks

## 🏗️ Project Structure

```
rag/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── config/                         # Configuration files
│   ├── naive_config.py            # Naive RAG settings
│   └── production_config.py       # Production RAG settings
│
├── core/                          # Core RAG components
│   ├── document_processor.py     # PDF parsing and chunking
│   ├── embeddings.py             # Embedding generation
│   ├── vector_store.py           # Vector storage (in-memory & ChromaDB)
│   └── retriever.py              # Retrieval strategies
│
├── pipelines/                     # RAG pipeline implementations
│   ├── naive_rag.py              # Naive RAG pipeline
│   └── production_rag.py         # Production RAG pipeline
│
├── visualizations/                # Visualization modules
│   ├── embeddings_viz.py         # Embedding plots (2D/3D)
│   └── similarity_viz.py         # Similarity score charts
│
├── utils/                        # Utility functions
│   └── helpers.py               # Helper functions
│
└── data/                        # Data directory (created at runtime)
    ├── uploads/                # Uploaded PDF files
    ├── vectorstore/           # Persistent vector database
    └── embedding_cache/       # Cached embeddings
```

## 🔍 Key Differences: Naive vs Production

| Feature | Naive RAG | Production RAG |
|---------|-----------|----------------|
| **Chunking** | Fixed 500 chars | Semantic 1000 chars + overlap |
| **Embeddings** | all-MiniLM-L6-v2 (384d) | all-mpnet-base-v2 (768d) |
| **Vector Store** | In-memory | ChromaDB (persistent) |
| **Retrieval** | Simple cosine | Hybrid + reranking |
| **Diversity** | ❌ | ✅ MMR |
| **Caching** | ❌ | ✅ Embeddings & queries |
| **Query Enhancement** | ❌ | ✅ Multi-query |
| **Best For** | Learning, prototypes | Production, accuracy |

## 🧪 Example Workflows

### Learning RAG Fundamentals
1. Start with **Naive RAG**
2. Upload a simple PDF (e.g., research paper, article)
3. Observe the chunking strategy
4. See how embeddings cluster in 2D space
5. Query and examine similarity scores

### Comparing Approaches
1. Process the same document with both modes
2. Use identical queries on both
3. Compare:
   - Chunking granularity
   - Retrieval quality (relevance scores)
   - Answer quality
   - Processing time

### Advanced Exploration
1. Use **Production RAG**
2. Upload complex documents
3. Explore reranking effects
4. Visualize query-document relationships in 3D
5. Experiment with different k values

## 🔧 Customization

### Modify Configurations

Edit configuration files to customize behavior:

**Naive RAG** ([naive_config.py](config/naive_config.py)):
```python
chunk_size = 500         # Adjust chunk size
top_k = 3               # Change retrieval count
embedding_model = "..."  # Use different model
```

**Production RAG** ([production_config.py](config/production_config.py)):
```python
chunk_size = 1000
chunk_overlap = 200
use_reranking = True
use_mmr = True
mmr_diversity_score = 0.3
```

### Add New Visualization Methods

Create new visualization functions in `visualizations/` directory and import them in `app.py`.

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Module not found" error
- **Solution**: Ensure virtual environment is activated and dependencies installed

**Issue**: ChromaDB persistence error
- **Solution**: Check write permissions for `rag/data/vectorstore/`

**Issue**: UMAP installation fails
- **Solution**: Try installing with `pip install umap-learn` separately

**Issue**: Slow embedding generation
- **Solution**: Use GPU if available or reduce chunk count

**Issue**: OpenAI API errors
- **Solution**: Check API key in `.env` file or use without LLM (context-only mode)

## 📚 Learning Resources

### Recommended Reading
- [RAG Fundamentals](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [Embedding Models Comparison](https://huggingface.co/blog/mteb)

### Next Steps
1. Experiment with different documents and queries
2. Try implementing new chunking strategies
3. Add support for other embedding providers (e.g., Cohere, Voyage)
4. Implement evaluation metrics (MRR, NDCG)
5. Add support for other document types (DOCX, TXT, Markdown)

## 🤝 Contributing

This is a learning project! Feel free to:
- Experiment with the code
- Add new features
- Improve visualizations
- Optimize performance
- Share your findings

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - Web framework
- [sentence-transformers](https://www.sbert.net/) - Embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Plotly](https://plotly.com/) - Visualizations
- [PyPDF](https://pypdf.readthedocs.io/) - PDF parsing

---

**Happy Learning! 🚀**

For questions or issues, please refer to the troubleshooting section or check the inline code documentation.
