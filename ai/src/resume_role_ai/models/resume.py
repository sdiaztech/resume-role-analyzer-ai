from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class Resume(BaseModel):
    """Normalized resume data used throughout the AI pipeline."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str = Field(..., min_length=1)
    raw_text: str = Field(..., min_length=1)
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    predicted_roles: list[str] = Field(default_factory=list)
