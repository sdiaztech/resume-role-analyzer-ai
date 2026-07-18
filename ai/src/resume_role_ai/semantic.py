from __future__ import annotations

import argparse
import hashlib
from typing import Any
from pathlib import Path

import joblib
import numpy as np
from resume_role_ai.paths import find_repository_root


DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbeddingRanker:
    """Ranks jobs using a pretrained sentence-transformer embedding model."""

    def __init__(
        self,
        job_texts: list[str],
        model_name: str = DEFAULT_SENTENCE_MODEL,
        encoder: Any | None = None,
    ):
        if len(job_texts) < 2:
            raise ValueError("At least two job profiles are required")
        self.model_name = model_name
        self.catalog_signature = catalog_signature(job_texts)
        self._encoder = encoder or self._load_encoder(model_name)
        self.job_embeddings = np.asarray(
            self._encoder.encode(
                job_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _load_encoder(model_name: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ValueError("Sentence-transformer support is not installed") from error
        return SentenceTransformer(model_name)

    def predict_scores(self, resume_text: str) -> np.ndarray:
        embedding = np.asarray(
            self._encoder.encode(
                [resume_text], normalize_embeddings=True, show_progress_bar=False
            )[0],
            dtype=np.float32,
        )
        return self.job_embeddings @ embedding

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "catalog_signature": self.catalog_signature,
                "job_embeddings": self.job_embeddings,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, job_texts: list[str]) -> SentenceEmbeddingRanker:
        artifact = joblib.load(path)
        if (
            not isinstance(artifact, dict)
            or artifact.get("catalog_signature") != catalog_signature(job_texts)
        ):
            raise ValueError("The sentence-embedding ranker does not match the current job catalog")
        ranker = cls.__new__(cls)
        ranker.model_name = artifact["model_name"]
        ranker.catalog_signature = artifact["catalog_signature"]
        ranker.job_embeddings = artifact["job_embeddings"]
        ranker._encoder = cls._load_encoder(ranker.model_name)
        return ranker


def rank_percentiles(scores: np.ndarray) -> np.ndarray:
    """Convert arbitrary scores to a comparable zero-to-one ranking scale."""
    if scores.ndim != 1 or not len(scores):
        raise ValueError("Scores must be a non-empty one-dimensional array")
    order = np.argsort(scores)
    percentiles = np.empty(len(scores), dtype=np.float32)
    percentiles[order] = np.linspace(0, 1, len(scores), dtype=np.float32)
    return percentiles


def catalog_signature(job_texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in job_texts:
        digest.update(text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    from resume_role_ai.predictor import RolePredictor, load_jobs

    root = find_repository_root()
    parser = argparse.ArgumentParser(description="Build the semantic job-catalog ranker")
    parser.add_argument(
        "--jobs", type=Path, default=root / "datasets" / "raw" / "job_positions.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "models" / "catalog_embeddings.joblib"
    )
    parser.add_argument("--model-name", default=DEFAULT_SENTENCE_MODEL)
    args = parser.parse_args()
    texts = [RolePredictor._job_text(job) for job in load_jobs(args.jobs)]
    SentenceEmbeddingRanker(texts, args.model_name).save(args.output)
    print(f"Wrote sentence-embedding ranker for {len(texts):,} profiles to {args.output}")


if __name__ == "__main__":
    main()
