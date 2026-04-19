"""
Visualization module for embeddings using dimensionality reduction.

Visualizes high-dimensional embeddings in 2D/3D space using:
- t-SNE
- UMAP
- PCA
"""

from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce embedding dimensions for visualization.

    Args:
        embeddings: High-dimensional embeddings (N x D)
        method: 'pca', 'tsne', or 'umap'
        n_components: Target dimensions (2 or 3)
        random_state: Random seed for reproducibility

    Returns:
        Reduced embeddings (N x n_components)
    """
    if len(embeddings) == 0:
        return np.array([])

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(embeddings)

    elif method == "tsne":
        # t-SNE can be slow for large datasets
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
        )
        return reducer.fit_transform(embeddings)

    elif method == "umap":
        try:
            from umap import UMAP
            n_neighbors = min(15, len(embeddings) - 1)
            reducer = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            raise ImportError(
                "UMAP not installed. Run: pip install umap-learn"
            )

    else:
        raise ValueError(f"Unknown method: {method}")


def create_embedding_plot_2d(
    embeddings: np.ndarray,
    texts: List[str],
    metadatas: List[dict],
    query_embedding: np.ndarray = None,
    query_text: str = None,
    method: str = "umap",
    title: str = "Document Embeddings Visualization",
) -> go.Figure:
    """
    Create an interactive 2D scatter plot of embeddings.

    Args:
        embeddings: Document embeddings
        texts: Corresponding text snippets
        metadatas: Metadata for each embedding
        query_embedding: Optional query embedding to highlight
        query_text: Query text for tooltip
        method: Dimensionality reduction method
        title: Plot title

    Returns:
        Plotly Figure object
    """
    if len(embeddings) == 0:
        # Return empty plot
        fig = go.Figure()
        fig.add_annotation(
            text="No embeddings to visualize",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Combine document and query embeddings for reduction
    all_embeddings = embeddings
    has_query = query_embedding is not None

    if has_query:
        all_embeddings = np.vstack([embeddings, query_embedding])

    # Reduce dimensions
    reduced = reduce_dimensions(all_embeddings, method=method, n_components=2)

    # Split back into documents and query
    doc_reduced = reduced[:len(embeddings)]
    query_reduced = reduced[-1:] if has_query else None

    # Truncate texts for hover
    hover_texts = [
        f"<b>Chunk {i}</b><br>{text[:200]}..."
        if len(text) > 200 else f"<b>Chunk {i}</b><br>{text}"
        for i, text in enumerate(texts)
    ]

    # Create figure
    fig = go.Figure()

    # Add document points
    fig.add_trace(go.Scatter(
        x=doc_reduced[:, 0],
        y=doc_reduced[:, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=range(len(doc_reduced)),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Chunk ID"),
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Document Chunks',
    ))

    # Add query point if provided
    if has_query and query_reduced is not None:
        fig.add_trace(go.Scatter(
            x=query_reduced[:, 0],
            y=query_reduced[:, 1],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star',
                line=dict(width=2, color='darkred'),
            ),
            text=[f"<b>Query</b><br>{query_text}"],
            hovertemplate='%{text}<extra></extra>',
            name='Query',
        ))

    fig.update_layout(
        title=title,
        xaxis_title=f"{method.upper()} Component 1",
        yaxis_title=f"{method.upper()} Component 2",
        hovermode='closest',
        template='plotly_white',
        height=600,
    )

    return fig


def create_embedding_plot_3d(
    embeddings: np.ndarray,
    texts: List[str],
    metadatas: List[dict],
    query_embedding: np.ndarray = None,
    query_text: str = None,
    method: str = "umap",
    title: str = "Document Embeddings Visualization (3D)",
) -> go.Figure:
    """
    Create an interactive 3D scatter plot of embeddings.

    Args:
        embeddings: Document embeddings
        texts: Corresponding text snippets
        metadatas: Metadata for each embedding
        query_embedding: Optional query embedding to highlight
        query_text: Query text for tooltip
        method: Dimensionality reduction method
        title: Plot title

    Returns:
        Plotly Figure object
    """
    if len(embeddings) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No embeddings to visualize",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Combine document and query embeddings
    all_embeddings = embeddings
    has_query = query_embedding is not None

    if has_query:
        all_embeddings = np.vstack([embeddings, query_embedding])

    # Reduce to 3D
    reduced = reduce_dimensions(all_embeddings, method=method, n_components=3)

    doc_reduced = reduced[:len(embeddings)]
    query_reduced = reduced[-1:] if has_query else None

    # Hover texts
    hover_texts = [
        f"<b>Chunk {i}</b><br>{text[:200]}..."
        if len(text) > 200 else f"<b>Chunk {i}</b><br>{text}"
        for i, text in enumerate(texts)
    ]

    # Create figure
    fig = go.Figure()

    # Add document points
    fig.add_trace(go.Scatter3d(
        x=doc_reduced[:, 0],
        y=doc_reduced[:, 1],
        z=doc_reduced[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=range(len(doc_reduced)),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Chunk ID"),
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Document Chunks',
    ))

    # Add query point
    if has_query and query_reduced is not None:
        fig.add_trace(go.Scatter3d(
            x=query_reduced[:, 0],
            y=query_reduced[:, 1],
            z=query_reduced[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='diamond',
                line=dict(width=2, color='darkred'),
            ),
            text=[f"<b>Query</b><br>{query_text}"],
            hovertemplate='%{text}<extra></extra>',
            name='Query',
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"{method.upper()} 1",
            yaxis_title=f"{method.upper()} 2",
            zaxis_title=f"{method.upper()} 3",
        ),
        hovermode='closest',
        template='plotly_white',
        height=700,
    )

    return fig
