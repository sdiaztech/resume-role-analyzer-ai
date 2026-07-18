from __future__ import annotations

from pydantic import BaseModel, Field


class JobPosition(BaseModel):
    """A job position that a resume can be matched against."""

    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    required_skills: list[str] = Field(default_factory=list)


class RoleMatch(BaseModel):
    job_id: str
    title: str
    score: float = Field(..., ge=0, le=1)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    explanation: str = Field(..., min_length=1)


class AnalysisResult(BaseModel):
    resume_id: str
    matches: list[RoleMatch]
