import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from backend.rag.retriever import FaissRetriever


def score_ranked_ids(ranked_ids, relevant_ids, k=3):
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("at least one relevant ID is required")

    ranked = list(ranked_ids)[:k]
    first_relevant_rank = next(
        (
            rank
            for rank, document_id in enumerate(ranked, start=1)
            if document_id in relevant
        ),
        None,
    )
    return {
        "cutoff": k,
        "hit_at_1": int(bool(ranked and ranked[0] in relevant)),
        "hit_at_k": int(any(document_id in relevant for document_id in ranked)),
        "reciprocal_rank_at_k": (
            1 / first_relevant_rank if first_relevant_rank else 0
        ),
        "first_relevant_rank": first_relevant_rank,
    }


def aggregate_scores(scores):
    if not scores:
        raise ValueError("at least one scored query is required")
    cutoff = scores[0]["cutoff"]
    if any(item["cutoff"] != cutoff for item in scores):
        raise ValueError("all scores must use the same cutoff")

    count = len(scores)
    return {
        "query_count": count,
        "hit_at_1": sum(item["hit_at_1"] for item in scores) / count,
        "hit_at_{}".format(cutoff): (
            sum(item["hit_at_k"] for item in scores) / count
        ),
        "mrr_at_{}".format(cutoff): (
            sum(item["reciprocal_rank_at_k"] for item in scores) / count
        ),
    }


def evaluate_queries(retriever, queries, top_k=3):
    results = []
    for case in queries:
        hits = retriever.search(case["question"], top_k)
        results.append(
            {
                "question": case["question"],
                "expected_ids": list(case["relevant_ids"]),
                "ranked_ids": [hit["id"] for hit in hits],
                "hits": [
                    {"id": hit["id"], "score": float(hit["score"])}
                    for hit in hits
                ],
                **score_ranked_ids(
                    [hit["id"] for hit in hits],
                    case["relevant_ids"],
                    top_k,
                ),
            }
        )
    return results


def main():
    data = json.loads(
        (HERE / "retrieval_cases.json").read_text(encoding="utf-8")
    )
    documents = data["documents"]
    queries = data["queries"]
    retriever = FaissRetriever(documents)
    results = evaluate_queries(retriever, queries, top_k=3)
    for case, result in zip(queries, results):
        print("\nQuestion:", case["question"])
        for hit in result["hits"]:
            print("  {}: {:.3f}".format(hit["id"], hit["score"]))
        print("Expected:", sorted(case["relevant_ids"]))
        print("First relevant rank:", result["first_relevant_rank"])

    metrics = aggregate_scores(results)
    print("\n--- RETRIEVAL REPORT ---")
    print("Queries:", metrics["query_count"])
    print("Hit@1:", round(metrics["hit_at_1"], 3))
    print("Hit@3:", round(metrics["hit_at_3"], 3))
    print("MRR@3:", round(metrics["mrr_at_3"], 3))


if __name__ == "__main__":
    main()
