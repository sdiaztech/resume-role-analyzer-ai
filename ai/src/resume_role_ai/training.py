from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from resume_role_ai.dataset import DEFAULT_OUTPUT as DEFAULT_PAIRS
from resume_role_ai.matcher import SupervisedMatcher
from resume_role_ai.paths import find_repository_root


DEFAULT_MODEL = find_repository_root() / "models" / "resume_matcher.joblib"
DEFAULT_METRICS = find_repository_root() / "models" / "metrics.json"
REQUIRED_COLUMNS = {"job_id", "job_text", "resume_text", "label", "split"}


def train_and_evaluate(
    pairs_path: str | Path = DEFAULT_PAIRS,
    model_path: str | Path = DEFAULT_MODEL,
    metrics_path: str | Path = DEFAULT_METRICS,
) -> dict[str, object]:
    pairs = pd.read_parquet(pairs_path)
    missing = REQUIRED_COLUMNS.difference(pairs.columns)
    if missing:
        raise ValueError(f"Training pairs are missing columns: {', '.join(sorted(missing))}")
    train = pairs[pairs["split"] == "train"]
    if train.empty:
        raise ValueError("The training split is empty")

    matcher = SupervisedMatcher.fit(train["job_text"], train["resume_text"], train["label"])
    matcher.save(model_path)
    metrics: dict[str, object] = {
        "model": "logistic-regression-pair-matcher",
        "training_pairs": len(train),
        "features": ["tfidf_cosine", "token_jaccard", "job_coverage", "length_ratio"],
    }
    for split_name in ("validation", "test"):
        split = pairs[pairs["split"] == split_name].copy()
        if split.empty:
            raise ValueError(f"The {split_name} split is empty")
        split["score"] = matcher.predict_scores(split["job_text"], split["resume_text"])
        metrics[split_name] = _evaluate_split(split)

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _evaluate_split(frame: pd.DataFrame) -> dict[str, float | int]:
    labels = frame["label"].to_numpy()
    scores = frame["score"].to_numpy()
    return {
        "pairs": len(frame),
        "accuracy": round(float(accuracy_score(labels, scores >= 0.5)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
        "average_precision": round(float(average_precision_score(labels, scores)), 4),
        "recall_at_5": round(_recall_at_k(frame, 5), 4),
        "recall_at_10": round(_recall_at_k(frame, 10), 4),
    }


def _recall_at_k(frame: pd.DataFrame, k: int) -> float:
    recalls: list[float] = []
    for _, group in frame.groupby("job_id"):
        positives = int(group["label"].sum())
        if positives:
            top = group.nlargest(k, "score")
            recalls.append(float(top["label"].sum()) / positives)
    return float(np.mean(recalls)) if recalls else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the resume/job matcher")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()
    metrics = train_and_evaluate(args.pairs, args.model, args.metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
