from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_role_ai.matcher import SupervisedMatcher
from resume_role_ai.models import AnalysisResult, JobPosition, Resume, RoleMatch
from resume_role_ai.paths import find_repository_root
from resume_role_ai.semantic import rank_percentiles
from resume_role_ai.skill_matching import normalized_resume_skills, skill_is_present


class SemanticRanker(Protocol):
    def predict_scores(self, resume_text: str) -> np.ndarray: ...


def load_jobs(path: str | Path | None = None) -> list[JobPosition]:
    """Load job positions from a CSV file."""
    path = path or find_repository_root() / "datasets" / "raw" / "job_positions.csv"
    with Path(path).open(encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        jobs = [
            JobPosition(
                id=row["id"],
                title=row["title"],
                description=row["description"],
                required_skills=[
                    skill.strip() for skill in row["required_skills"].split("|") if skill.strip()
                ],
            )
            for row in rows
        ]
    if not jobs:
        raise ValueError("The job-position database is empty")
    return jobs


class RolePredictor:
    """Ranks known positions using TF-IDF text similarity and explicit skill overlap."""

    def __init__(
        self,
        jobs: list[JobPosition],
        matcher: SupervisedMatcher | None = None,
        semantic_ranker: SemanticRanker | None = None,
        supervised_weight: float = 0.0,
    ):
        if not jobs:
            raise ValueError("At least one job position is required")
        self.jobs = jobs
        self.matcher = matcher
        self.semantic_ranker = semantic_ranker
        self.supervised_weight = supervised_weight
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._job_matrix = self.vectorizer.fit_transform(self._job_text(job) for job in jobs)
        self._skill_weights = self._calculate_skill_weights(jobs)

    @staticmethod
    def _calculate_skill_weights(jobs: list[JobPosition]) -> dict[str, float]:
        """Give rare, career-specific skills more influence than generic skills."""
        document_frequency: dict[str, int] = {}
        for job in jobs:
            for skill in {skill.casefold() for skill in job.required_skills}:
                document_frequency[skill] = document_frequency.get(skill, 0) + 1
        return {
            skill: math.log((len(jobs) + 1) / (frequency + 1)) + 1
            for skill, frequency in document_frequency.items()
        }

    @staticmethod
    def _career_family(job: JobPosition) -> str:
        """Use the two-digit SOC group as the broad family for O*NET occupations."""
        match = re.fullmatch(r"onet-(\d{2})-\d{4}-\d{2}", job.id)
        # Hand-authored roles are kept independent instead of becoming one fake family.
        return match.group(1) if match else f"custom:{job.id}"

    def _weighted_skill_coverage(self, job: JobPosition, resume_skills: set[str]) -> float:
        weights = [self._skill_weights.get(skill.casefold(), 1.0) for skill in job.required_skills]
        if not weights:
            return 0.0
        matched_weight = sum(
            weight
            for skill, weight in zip(job.required_skills, weights, strict=True)
            if skill_is_present(skill, resume_skills)
        )
        return matched_weight / sum(weights)

    @staticmethod
    def _partition_skills(
        job: JobPosition, resume_skills: set[str]
    ) -> tuple[list[str], list[str]]:
        matched: list[str] = []
        missing: list[str] = []
        for skill in job.required_skills:
            (matched if skill_is_present(skill, resume_skills) else missing).append(skill)
        return matched, missing

    def _family_candidates(
        self, scores: np.ndarray, result_limit: int, family_limit: int = 3
    ) -> np.ndarray:
        """Keep occupations from the strongest broad career families."""
        family_scores: dict[str, float] = {}
        for index, job in enumerate(self.jobs):
            family = self._career_family(job)
            family_scores[family] = max(family_scores.get(family, float("-inf")), scores[index])
        ranked_families = sorted(
            family_scores.items(), key=lambda item: item[1], reverse=True
        )
        selected: set[str] = set()
        candidate_count = 0
        for family, _ in ranked_families:
            selected.add(family)
            candidate_count += sum(
                self._career_family(job) == family for job in self.jobs
            )
            if len(selected) >= family_limit and candidate_count >= result_limit:
                break
        return np.array(
            [index for index, job in enumerate(self.jobs) if self._career_family(job) in selected]
        )

    @staticmethod
    def _job_text(job: JobPosition) -> str:
        # Repeat structured skills so that they carry more weight than generic description words.
        skills = " ".join(job.required_skills)
        return f"{job.title} {job.description} {skills} {skills}"

    @staticmethod
    def _resume_text(resume: Resume) -> str:
        structured = " ".join(
            resume.skills + resume.education + resume.experience + resume.certifications
        )
        return f"{resume.raw_text} {structured} {structured}"

    def predict(self, resume: Resume, limit: int = 3) -> AnalysisResult:
        limit = max(1, min(limit, len(self.jobs)))
        resume_text = self._resume_text(resume)
        supervised_scores = None
        if self.matcher is not None:
            supervised_scores = self.matcher.predict_scores(
                [self._job_text(job) for job in self.jobs],
                [resume_text] * len(self.jobs),
            )
            similarities = supervised_scores
        else:
            resume_vector = self.vectorizer.transform([resume_text])
            similarities = cosine_similarity(resume_vector, self._job_matrix)[0]
        display_scores = similarities
        if self.semantic_ranker is not None:
            semantic_raw = self.semantic_ranker.predict_scores(resume_text)
            semantic_scores = rank_percentiles(semantic_raw)
            similarities = (
                self.supervised_weight * rank_percentiles(supervised_scores)
                + (1 - self.supervised_weight) * semantic_scores
                if supervised_scores is not None
                else semantic_scores
            )
            display_scores = (
                self.supervised_weight * supervised_scores
                + (1 - self.supervised_weight) * np.clip(semantic_raw, 0, 1)
                if supervised_scores is not None
                else np.clip(semantic_raw, 0, 1)
            )
        resume_skills = normalized_resume_skills(resume.skills)
        skill_scores = np.array(
            [self._weighted_skill_coverage(job, resume_skills) for job in self.jobs]
        )
        ranking_scores = 0.85 * rank_percentiles(similarities) + 0.15 * skill_scores
        candidate_indexes = self._family_candidates(ranking_scores, limit)

        matches: list[RoleMatch] = []
        ranked_indexes = candidate_indexes[np.argsort(ranking_scores[candidate_indexes])[::-1]]
        for index in ranked_indexes[:limit]:
            job = self.jobs[index]
            matched, missing = self._partition_skills(job, resume_skills)
            overlap = self._weighted_skill_coverage(job, resume_skills)
            score = (
                float(display_scores[index])
                if self.matcher is not None or self.semantic_ranker is not None
                else 0.8 * float(similarities[index]) + 0.2 * overlap
            )
            matches.append(
                RoleMatch(
                    job_id=job.id,
                    title=job.title,
                    score=round(min(score, 1.0), 4),
                    matched_skills=matched,
                    missing_skills=missing,
                    explanation=self._explain_match(job, matched, missing),
                )
            )
        return AnalysisResult(resume_id=resume.id, matches=matches)

    @staticmethod
    def _explain_match(
        job: JobPosition, matched_skills: list[str], missing_skills: list[str]
    ) -> str:
        if matched_skills:
            strengths = ", ".join(matched_skills[:3])
            explanation = f"Your resume aligns with {job.title} through {strengths}."
        else:
            explanation = f"Your resume text is related to work commonly done by a {job.title}."
        if missing_skills:
            explanation += f" Strengthen the match by developing {', '.join(missing_skills[:3])}."
        return explanation
