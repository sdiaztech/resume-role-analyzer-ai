from __future__ import annotations

import argparse
import hashlib
import random
import urllib.request
from pathlib import Path

import pandas as pd

from resume_role_ai.paths import find_repository_root


DATASET_ID = "michaelozon/candidate-matching-synthetic"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"
FILES = {
    "resumes.parquet": (
        "resumes/train-00000-of-00001.parquet",
        "4a25add9c3a620c4c396e867063e390fc8084f9b53ed8135def049303a29d7a4",
    ),
    "jobs.parquet": (
        "jobs/train-00000-of-00001.parquet",
        "c13d7c7e0ab510e81117974027de1364f973a27b3cacc8fabfbd8851c352c871",
    ),
    "matches.parquet": (
        "matches/train-00000-of-00001.parquet",
        "dab24e40948b2da348c6df7072b5b5fe3bb6ef8cab335f88eef5d5667ee79a92",
    ),
}
REPOSITORY_ROOT = find_repository_root()
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "datasets" / "external" / "candidate-matching-synthetic"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "datasets" / "processed" / "training_pairs.parquet"


def download_dataset(destination: str | Path = DEFAULT_SOURCE_DIR) -> Path:
    """Download and verify the three MIT-licensed source tables."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, (remote_path, expected_hash) in FILES.items():
        target = destination / filename
        if target.exists() and _sha256(target) == expected_hash:
            continue
        temporary = target.with_suffix(f"{target.suffix}.part")
        try:
            urllib.request.urlretrieve(f"{BASE_URL}/{remote_path}?download=true", temporary)
            if _sha256(temporary) != expected_hash:
                raise ValueError(f"Checksum verification failed for {filename}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_training_pairs(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    negatives_per_positive: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Create labeled resume/job pairs with deterministic negative sampling."""
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be at least 1")

    source_dir = Path(source_dir)
    resumes = pd.read_parquet(source_dir / "resumes.parquet")
    jobs = pd.read_parquet(source_dir / "jobs.parquet")
    matches = pd.read_parquet(source_dir / "matches.parquet")
    _require_columns(resumes, {"resume_id", "summary", "skills", "experience_bullets"})
    _require_columns(
        jobs, {"job_id", "job_title", "description", "must_have_skills", "requirements"}
    )
    _require_columns(matches, {"job_id", "relevant_resume_ids"})

    resume_text = {
        row.resume_id: _join_text(row.summary, row.skills, row.experience_bullets)
        for row in resumes.itertuples(index=False)
    }
    job_text = {
        row.job_id: _join_text(
            row.job_title, row.description, row.must_have_skills, row.requirements
        )
        for row in jobs.itertuples(index=False)
    }
    resume_ids = sorted(resume_text)
    rng = random.Random(seed)
    records: list[dict[str, object]] = []

    for row in matches.sort_values("job_id").itertuples(index=False):
        positives = set(row.relevant_resume_ids)
        if row.job_id not in job_text:
            raise ValueError(f"Match references unknown job {row.job_id}")
        for resume_id in sorted(positives):
            if resume_id not in resume_text:
                raise ValueError(f"Match references unknown resume {resume_id}")
            records.append(_pair(row.job_id, resume_id, job_text, resume_text, 1))

        negative_candidates = [item for item in resume_ids if item not in positives]
        count = min(len(positives) * negatives_per_positive, len(negative_candidates))
        for resume_id in rng.sample(negative_candidates, count):
            records.append(_pair(row.job_id, resume_id, job_text, resume_text, 0))

    pairs = pd.DataFrame.from_records(records)
    pairs["split"] = pairs["job_id"].map(_split_for_job)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(output, index=False)
    return pairs


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")


def _join_text(*values: object) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, str):
            parts.append(value)
        else:
            parts.extend(str(item) for item in value)
    return " ".join(parts)


def _pair(
    job_id: str,
    resume_id: str,
    jobs: dict[str, str],
    resumes: dict[str, str],
    label: int,
) -> dict[str, object]:
    return {
        "job_id": job_id,
        "resume_id": resume_id,
        "job_text": jobs[job_id],
        "resume_text": resumes[resume_id],
        "label": label,
    }


def _split_for_job(job_id: str) -> str:
    bucket = int(hashlib.sha256(job_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare resume/job training data")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    args = parser.parse_args()
    if not args.skip_download:
        download_dataset(args.source_dir)
    pairs = prepare_training_pairs(
        args.source_dir,
        args.output,
        negatives_per_positive=args.negatives_per_positive,
    )
    counts = pairs.groupby(["split", "label"]).size().to_dict()
    print(f"Wrote {len(pairs):,} labeled pairs to {args.output}")
    print(f"Split/label counts: {counts}")


if __name__ == "__main__":
    main()
