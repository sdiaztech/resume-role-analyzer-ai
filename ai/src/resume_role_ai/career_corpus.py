from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from resume_role_ai.matcher import SupervisedMatcher
from resume_role_ai.paths import find_repository_root
from resume_role_ai.predictor import RolePredictor, load_jobs
from resume_role_ai.semantic import SentenceEmbeddingRanker, rank_percentiles


DATASET_ID = "wzzwn37gmd"
DATASET_DOI = "10.17632/wzzwn37gmd.1"
FILE_ID = "8dd12325-8dd9-47ea-9b90-8cf1e84eaf2a"
FILE_SHA256 = "97b1fa6cca1232912dd3f7bc312d71a2478249787168fd41dfbf6c1da90479d3"
DOWNLOAD_URL = (
    f"https://data.mendeley.com/public-files/datasets/{DATASET_ID}/files/"
    f"{FILE_ID}/file_downloaded"
)
REPOSITORY_ROOT = find_repository_root()
DEFAULT_WORKBOOK = REPOSITORY_ROOT / "datasets" / "external" / "career-corpus" / "CareerCorpus.xlsx"
DEFAULT_CATALOG = REPOSITORY_ROOT / "datasets" / "raw" / "job_positions.csv"
DEFAULT_MODEL = REPOSITORY_ROOT / "models" / "resume_matcher.joblib"
DEFAULT_REPORT = REPOSITORY_ROOT / "models" / "career_corpus_metrics.json"
REQUIRED_COLUMNS = {
    "ID",
    "Domain",
    "Education",
    "Skills and Achievements",
    "Experience",
    "Annotator-1",
    "Annotator-2",
}
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def download_career_corpus(destination: str | Path = DEFAULT_WORKBOOK) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _sha256(destination) == FILE_SHA256:
        return destination
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, temporary)
        if _sha256(temporary) != FILE_SHA256:
            raise ValueError("CareerCorpus checksum verification failed")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def evaluate_career_corpus(
    workbook: str | Path = DEFAULT_WORKBOOK,
    catalog: str | Path = DEFAULT_CATALOG,
    model: str | Path = DEFAULT_MODEL,
    report: str | Path = DEFAULT_REPORT,
    *,
    batch_size: int = 16,
) -> dict[str, object]:
    frame = pd.read_excel(workbook).fillna("")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"CareerCorpus is missing columns: {', '.join(sorted(missing))}")
    jobs = load_jobs(catalog)
    matcher = SupervisedMatcher.load(model)
    job_texts = [RolePredictor._job_text(job) for job in jobs]
    semantic_ranker = SentenceEmbeddingRanker(job_texts)
    predictor = RolePredictor(jobs)
    expected_by_domain = {
        domain: _expected_job_indices(str(domain), [job.title for job in jobs])
        for domain in sorted(frame["Domain"].astype(str).unique())
    }
    empty = [domain for domain, indices in expected_by_domain.items() if not indices]
    if empty:
        raise ValueError(f"No catalog occupations match domains: {', '.join(empty)}")

    results: list[dict[str, object]] = []
    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[start : start + batch_size]
        resume_texts = [_resume_text(row) for _, row in batch.iterrows()]
        scores = matcher.predict_scores(
            job_texts * len(resume_texts),
            [text for text in resume_texts for _ in job_texts],
        ).reshape(len(resume_texts), len(job_texts))
        for row_index, (_, row) in enumerate(batch.iterrows()):
            expected = expected_by_domain[str(row["Domain"])]
            supervised = rank_percentiles(scores[row_index])
            semantic = rank_percentiles(semantic_ranker.predict_scores(resume_texts[row_index]))
            ranks = {}
            for weight in BLEND_WEIGHTS:
                combined = weight * supervised + (1 - weight) * semantic
                candidates = predictor._family_candidates(combined, result_limit=10)
                ranks[_weight_key(weight)] = _first_expected_rank(
                    combined, expected, candidates
                )
            results.append(
                {
                    "domain": str(row["Domain"]),
                    **ranks,
                    "evaluation_split": _evaluation_split(str(row["ID"])),
                    "annotation_score": (
                        float(row["Annotator-1"]) + float(row["Annotator-2"])
                    )
                    / 2,
                }
            )

    result_frame = pd.DataFrame(results)
    validation = result_frame[result_frame["evaluation_split"] == "validation"]
    selected_weight = max(
        BLEND_WEIGHTS,
        key=lambda weight: (
            _ranking_metrics(validation, _weight_key(weight))["hit_at_10"],
            _ranking_metrics(validation, _weight_key(weight))["mean_reciprocal_rank"],
        ),
    )
    selected_column = _weight_key(selected_weight)
    metrics: dict[str, object] = {
        "dataset": "CareerCorpus",
        "doi": DATASET_DOI,
        "samples": len(result_frame),
        "catalog_profiles": len(jobs),
        "mean_expert_annotation": round(float(result_frame["annotation_score"].mean()), 4),
        "selected_supervised_weight": selected_weight,
        "overall": _ranking_metrics(result_frame, selected_column),
        "validation": _ranking_metrics(validation, selected_column),
        "test": _ranking_metrics(
            result_frame[result_frame["evaluation_split"] == "test"], selected_column
        ),
        "strategy_comparison": {
            _strategy_name(weight): _ranking_metrics(result_frame, _weight_key(weight))
            for weight in BLEND_WEIGHTS
        },
        "by_domain": {
            domain: _ranking_metrics(group, selected_column)
            for domain, group in result_frame.groupby("domain", sort=True)
        },
    }
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _resume_text(row: pd.Series) -> str:
    return " ".join(
        str(row[column])
        for column in ("Education", "Skills and Achievements", "Experience")
    )


def _expected_job_indices(domain: str, titles: list[str]) -> set[int]:
    predicates = {
        "accountant": ("accountant", "auditor"),
        "apparel": ("apparel", "fashion designer", "tailor", "dressmaker", "sewing"),
        "banking": ("bank teller", "teller", "loan officer", "new accounts clerk"),
        "finance": ("financial", "finance"),
        "research assistant": ("research assistant",),
        "teacher": ("teacher", "instructor"),
    }
    keywords = predicates.get(domain.casefold())
    if keywords is None:
        raise ValueError(f"Unsupported CareerCorpus domain: {domain}")
    return {
        index
        for index, title in enumerate(titles)
        if any(keyword in title.casefold() for keyword in keywords)
    }


def _first_expected_rank(
    scores: np.ndarray, expected: set[int], candidates: np.ndarray | None = None
) -> int:
    candidates = candidates if candidates is not None else np.arange(len(scores))
    ranking = candidates[np.argsort(scores[candidates])[::-1]]
    return next(
        (rank for rank, job_index in enumerate(ranking, start=1) if job_index in expected),
        len(scores) + 1,
    )


def _evaluation_split(resume_id: str) -> str:
    bucket = int(hashlib.sha256(resume_id.encode()).hexdigest()[:8], 16) % 2
    return "validation" if bucket == 0 else "test"


def _weight_key(weight: float) -> str:
    return f"rank_{weight:.2f}"


def _strategy_name(weight: float) -> str:
    if weight == 0:
        return "semantic"
    if weight == 1:
        return "supervised"
    return f"hybrid_{weight:.2f}_supervised"


def _ranking_metrics(frame: pd.DataFrame, rank_column: str = "rank") -> dict[str, float | int]:
    ranks = frame[rank_column]
    return {
        "samples": len(frame),
        "hit_at_1": round(float((ranks <= 1).mean()), 4),
        "hit_at_5": round(float((ranks <= 5).mean()), 4),
        "hit_at_10": round(float((ranks <= 10).mean()), 4),
        "mean_reciprocal_rank": round(float((1 / ranks).mean()), 4),
        "median_rank": int(ranks.median()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate matching with CareerCorpus")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if not args.skip_download:
        download_career_corpus(args.workbook)
    metrics = evaluate_career_corpus(args.workbook, args.catalog, args.model, args.report)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
