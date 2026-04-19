"""
Visualization module for similarity scores and retrieval results.

Shows:
- Similarity score distributions
- Top-k retrieval results
- Score comparisons between naive and production RAG
"""

from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rag.core.vector_store import SearchResult


def create_similarity_bar_chart(
    results: List[SearchResult],
    title: str = "Retrieval Results - Similarity Scores",
    max_results: int = 10,
) -> go.Figure:
    """
    Create a bar chart showing similarity scores for retrieved chunks.

    Args:
        results: List of SearchResult objects
        title: Chart title
        max_results: Maximum number of results to display

    Returns:
        Plotly Figure object
    """
    if not results:
        fig = go.Figure()
        fig.add_annotation(
            text="No results to visualize",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Limit results
    results = results[:max_results]

    # Prepare data
    chunk_labels = [f"Chunk {r.chunk_id}" for r in results]
    scores = [r.score for r in results]
    texts = [r.text[:100] + "..." if len(r.text) > 100 else r.text for r in results]

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=chunk_labels,
        y=scores,
        text=[f"{s:.3f}" for s in scores],
        textposition='auto',
        marker=dict(
            color=scores,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Score"),
        ),
        hovertext=texts,
        hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<br><br>%{hovertext}<extra></extra>',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Retrieved Chunks",
        yaxis_title="Similarity Score",
        template='plotly_white',
        height=500,
    )

    return fig


def create_comparison_chart(
    naive_results: List[SearchResult],
    production_results: List[SearchResult],
    title: str = "Naive vs Production RAG - Score Comparison",
) -> go.Figure:
    """
    Create a comparison chart between naive and production RAG results.

    Args:
        naive_results: Results from naive RAG
        production_results: Results from production RAG
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Get common chunk IDs
    naive_scores = {r.chunk_id: r.score for r in naive_results}
    prod_scores = {r.chunk_id: r.score for r in production_results}

    all_chunk_ids = sorted(set(naive_scores.keys()) | set(prod_scores.keys()))

    # Prepare data
    chunk_labels = [f"Chunk {cid}" for cid in all_chunk_ids]
    naive_vals = [naive_scores.get(cid, 0) for cid in all_chunk_ids]
    prod_vals = [prod_scores.get(cid, 0) for cid in all_chunk_ids]

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Naive RAG',
        x=chunk_labels,
        y=naive_vals,
        marker_color='lightblue',
    ))

    fig.add_trace(go.Bar(
        name='Production RAG',
        x=chunk_labels,
        y=prod_vals,
        marker_color='darkblue',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Chunks",
        yaxis_title="Similarity Score",
        barmode='group',
        template='plotly_white',
        height=500,
    )

    return fig


def create_retrieval_heatmap(
    results_list: List[List[SearchResult]],
    query_labels: List[str],
    title: str = "Retrieval Scores Heatmap",
) -> go.Figure:
    """
    Create a heatmap showing retrieval scores across multiple queries.

    Args:
        results_list: List of result lists (one per query)
        query_labels: Labels for each query
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Find all unique chunk IDs
    all_chunk_ids = set()
    for results in results_list:
        all_chunk_ids.update(r.chunk_id for r in results)

    chunk_ids = sorted(all_chunk_ids)

    # Build score matrix
    score_matrix = []
    for results in results_list:
        scores_dict = {r.chunk_id: r.score for r in results}
        row = [scores_dict.get(cid, 0) for cid in chunk_ids]
        score_matrix.append(row)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=score_matrix,
        x=[f"Chunk {cid}" for cid in chunk_ids],
        y=query_labels,
        colorscale='Blues',
        hoverongaps=False,
        colorbar=dict(title="Score"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Chunks",
        yaxis_title="Queries",
        template='plotly_white',
        height=400,
    )

    return fig


def create_score_distribution(
    results: List[SearchResult],
    title: str = "Similarity Score Distribution",
) -> go.Figure:
    """
    Create a histogram showing the distribution of similarity scores.

    Args:
        results: List of SearchResult objects
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if not results:
        fig = go.Figure()
        fig.add_annotation(
            text="No results to visualize",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    scores = [r.score for r in results]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker_color='steelblue',
        opacity=0.7,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Similarity Score",
        yaxis_title="Frequency",
        template='plotly_white',
        height=400,
    )

    return fig
