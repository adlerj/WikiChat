"""Reciprocal Rank Fusion for hybrid retrieval."""
from typing import List, Dict


def rrf_score(rank: int, k: int = 60) -> float:
    """Compute RRF score.

    Args:
        rank: Rank position (0-indexed)
        k: RRF constant

    Returns:
        RRF score
    """
    return 1.0 / (k + rank)


def rrf_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = 60,
) -> List[Dict]:
    """Fuse dense and sparse results using RRF.

    Args:
        dense_results: Results from dense retrieval
        sparse_results: Results from sparse retrieval
        k: RRF constant

    Returns:
        Fused and ranked results
    """
    # Collect scores for each chunk
    chunk_scores = {}

    # Process sparse first (convention: sparse gets priority in ties)
    for result in sparse_results:
        chunk_id = result["chunk_id"]
        rank = result["rank"]
        score = rrf_score(rank, k)
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + score

    for result in dense_results:
        chunk_id = result["chunk_id"]
        rank = result["rank"]
        score = rrf_score(rank, k)
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + score

    # Sort by score (descending), then by chunk_id (descending) for determinism
    sorted_chunks = sorted(
        chunk_scores.items(),
        key=lambda x: (-x[1], -int(x[0]) if x[0].isdigit() else x[0]),
    )

    # Format results
    results = []
    for rank, (chunk_id, score) in enumerate(sorted_chunks):
        results.append({
            "chunk_id": chunk_id,
            "score": score,
            "rank": rank,
        })

    return results
