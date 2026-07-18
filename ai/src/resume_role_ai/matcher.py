from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


MODEL_VERSION = 1
TOKEN_PATTERN = re.compile(r"[a-z0-9+#.]+")


@dataclass
class SupervisedMatcher:
    """A trained binary classifier that scores resume/job relevance."""

    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    version: int = MODEL_VERSION

    @classmethod
    def fit(
        cls,
        job_texts: Iterable[str],
        resume_texts: Iterable[str],
        labels: Iterable[int],
    ) -> SupervisedMatcher:
        jobs = list(job_texts)
        resumes = list(resume_texts)
        targets = np.asarray(list(labels), dtype=np.int8)
        if not jobs or len(jobs) != len(resumes) or len(jobs) != len(targets):
            raise ValueError("Training jobs, resumes, and labels must have equal non-zero lengths")
        if set(targets) != {0, 1}:
            raise ValueError("Training labels must contain both 0 and 1")

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=30_000,
            min_df=2,
            dtype=np.float32,
        )
        vectorizer.fit([*jobs, *resumes])
        features = pair_features(vectorizer, jobs, resumes)
        classifier = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        classifier.fit(features, targets)
        return cls(vectorizer=vectorizer, classifier=classifier)

    def predict_scores(
        self, job_texts: Iterable[str], resume_texts: Iterable[str]
    ) -> np.ndarray:
        jobs = list(job_texts)
        resumes = list(resume_texts)
        if len(jobs) != len(resumes):
            raise ValueError("Jobs and resumes must have equal lengths")
        return self.classifier.predict_proba(pair_features(self.vectorizer, jobs, resumes))[:, 1]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> SupervisedMatcher:
        matcher = joblib.load(path)
        if not isinstance(matcher, cls) or matcher.version != MODEL_VERSION:
            raise ValueError("The matcher artifact is incompatible with this application version")
        return matcher


def pair_features(
    vectorizer: TfidfVectorizer,
    job_texts: list[str],
    resume_texts: list[str],
) -> np.ndarray:
    job_vectors = vectorizer.transform(job_texts)
    resume_vectors = vectorizer.transform(resume_texts)
    cosine = np.asarray(job_vectors.multiply(resume_vectors).sum(axis=1)).ravel()

    overlaps = np.empty((len(job_texts), 3), dtype=np.float32)
    for index, (job, resume) in enumerate(zip(job_texts, resume_texts, strict=True)):
        job_tokens = set(TOKEN_PATTERN.findall(job.casefold()))
        resume_tokens = set(TOKEN_PATTERN.findall(resume.casefold()))
        intersection = len(job_tokens & resume_tokens)
        union = len(job_tokens | resume_tokens)
        overlaps[index] = (
            intersection / union if union else 0,
            intersection / len(job_tokens) if job_tokens else 0,
            min(len(job_tokens), len(resume_tokens)) / max(len(job_tokens), len(resume_tokens), 1),
        )
    return np.column_stack((cosine, overlaps))
