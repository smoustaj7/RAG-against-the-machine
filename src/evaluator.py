"""Evaluation module for RAG retrieval quality.

Computes recall@k metrics by comparing retrieved sources against
ground-truth sources using IoU-based character-range overlap.
"""

from typing import Dict, List
from uuid import UUID

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def source_overlaps(
    retrieved: MinimalSource,
    ground_truth: MinimalSource,
    iou_threshold: float = 0.05,
) -> bool:
    """Check whether a retrieved source overlaps a ground-truth source.

    Two sources overlap if they share the same file_path AND the
    Intersection-over-Union (IoU) of their character ranges is at
    least ``iou_threshold``.

    Args:
        retrieved: the student-retrieved source.
        ground_truth: the ground-truth source from the dataset.
        iou_threshold: minimum IoU for a match (default 0.05).

    Returns:
        True if the sources overlap sufficiently.
    """
    if retrieved.file_path != ground_truth.file_path:
        return False

    r_start = retrieved.first_character_index
    r_end = retrieved.last_character_index
    g_start = ground_truth.first_character_index
    g_end = ground_truth.last_character_index

    inter_start = max(r_start, g_start)
    inter_end = min(r_end, g_end)
    intersection = max(0, inter_end - inter_start)

    union = (r_end - r_start) + (g_end - g_start) - intersection
    if union <= 0:
        return False

    iou = intersection / union
    return iou >= iou_threshold


def question_hit_at_k(
    retrieved_sources: List[MinimalSource],
    gt_sources: List[MinimalSource],
    k: int,
) -> bool:
    """Determine if a question is a hit at rank k.

    A question is a hit if *any* of the top-k retrieved sources
    overlaps with *any* of the ground-truth sources.

    Args:
        retrieved_sources: ranked list of retrieved sources.
        gt_sources: ground-truth sources for this question.
        k: cutoff rank.

    Returns:
        True if at least one retrieved source (within top-k)
        overlaps with at least one ground-truth source.
    """
    top_k = retrieved_sources[:k]
    for ret_src in top_k:
        for gt_src in gt_sources:
            if source_overlaps(ret_src, gt_src):
                return True
    return False


def evaluate_search_results(
    student_results: StudentSearchResults,
    dataset: RagDataset,
) -> Dict[str, float]:
    """Evaluate student search results against a ground-truth dataset.

    Computes recall@1, recall@3, recall@5, and recall@10.

    Args:
        student_results: the student's ``StudentSearchResults``.
        dataset: the ground-truth ``RagDataset`` of
            ``AnsweredQuestion`` items.

    Returns:
        Dictionary with keys ``recall@1``, ``recall@3``,
        ``recall@5``, ``recall@10`` and their float values.
    """
    gt_by_id: Dict[UUID, AnsweredQuestion] = {}
    for q in dataset.rag_questions:
        if isinstance(q, AnsweredQuestion):
            gt_by_id[q.question_id] = q

    if not gt_by_id:
        return {
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
        }

    k_values = [1, 3, 5, 10]
    hits: Dict[int, int] = {k: 0 for k in k_values}
    matched = 0

    for sr in student_results.search_results:
        gt_q = gt_by_id.get(sr.question_id)
        if gt_q is None:
            continue

        matched += 1
        for k in k_values:
            if question_hit_at_k(sr.retrieved_sources, gt_q.sources, k):
                hits[k] += 1

    total = matched if matched > 0 else 1

    return {
        f"recall@{k}": hits[k] / total for k in k_values
    }
