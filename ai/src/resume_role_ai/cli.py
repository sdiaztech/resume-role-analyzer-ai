from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from resume_role_ai.matcher import SupervisedMatcher
from resume_role_ai.models import Resume
from resume_role_ai.paths import find_repository_root
from resume_role_ai.parser import parse_resume
from resume_role_ai.predictor import RolePredictor, load_jobs
from resume_role_ai.semantic import SentenceEmbeddingRanker


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict suitable roles for a resume")
    parser.add_argument("--jobs", help="Path to the job-position CSV")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--file", help="PDF, DOCX, or TXT resume to parse")
    parser.add_argument("--id", default="uploaded-resume", help="ID used with --file")
    parser.add_argument(
        "--model",
        type=Path,
        default=find_repository_root() / "models" / "resume_matcher.joblib",
        help="Trained matcher artifact; the TF-IDF baseline is used when it does not exist",
    )
    parser.add_argument(
        "--semantic-model",
        type=Path,
        default=find_repository_root() / "models" / "catalog_embeddings.joblib",
        help="Semantic catalog artifact; ignored when it is absent or stale",
    )
    args = parser.parse_args()

    try:
        jobs = load_jobs(args.jobs) if args.jobs else load_jobs()
        resume = (
            parse_resume(args.file, args.id, jobs)
            if args.file
            else Resume.model_validate_json(sys.stdin.read())
        )
        matcher = SupervisedMatcher.load(args.model) if args.model.is_file() else None
        job_texts = [RolePredictor._job_text(job) for job in jobs]
        try:
            semantic = (
                SentenceEmbeddingRanker.load(args.semantic_model, job_texts)
                if args.semantic_model.is_file()
                else None
            )
        except ValueError:
            semantic = None
        predictor = RolePredictor(jobs, matcher, semantic)
        print(predictor.predict(resume, args.limit).model_dump_json())
    except (ValidationError, ValueError, OSError, KeyError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
