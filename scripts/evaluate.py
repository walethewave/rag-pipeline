"""Measures retrieval quality: for a list of (question, expected_chunk_question) pairs, checks
whether the expected chunk comes back in the top-k results, and reports hit rate@k and MRR.

Run with: python scripts/evaluate.py [questions.json] [--index faq] [--k 3]
Expects the FAQ index to already be populated (run scripts/make_sample_pdf.py + upload it first,
or point --index at an index you already ingested documents into).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai import get_embedder  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.main import _retrieve  # noqa: E402

DEFAULT_QUESTIONS_PATH = Path(__file__).parent / "sample_eval_questions.json"


def evaluate(questions: list[dict], index: str, k: int) -> dict:
    """Runs retrieval for every question and scores it.

    hit@k: fraction of questions where the expected chunk's question text appears anywhere in
    the top-k retrieved results (a simple substring check, good enough for a teaching script).

    MRR (Mean Reciprocal Rank): the average of 1/rank of the first correct hit (0 if it never
    appears in the top-k), rewarding the expected chunk landing higher in the results.
    """
    embedder = get_embedder()
    settings = get_settings()
    hits = 0
    reciprocal_ranks = []

    for item in questions:
        question, expected = item["question"], item["expected_chunk_question"]
        chunks, _, _ = _retrieve(question, index, k, embedder, settings)

        rank = next(
            (i for i, c in enumerate(chunks, start=1) if expected.lower() in c.text.lower()),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(questions)
    return {
        "total_questions": total,
        "hit_rate_at_k": round(hits / total, 3) if total else 0.0,
        "mrr": round(sum(reciprocal_ranks) / total, 3) if total else 0.0,
        "k": k,
        "index": index,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions_file", nargs="?", default=str(DEFAULT_QUESTIONS_PATH))
    parser.add_argument("--index", default="faq")
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()

    with open(args.questions_file, encoding="utf-8") as f:
        questions = json.load(f)

    results = evaluate(questions, index=args.index, k=args.k)
    print(json.dumps(results, indent=2))
