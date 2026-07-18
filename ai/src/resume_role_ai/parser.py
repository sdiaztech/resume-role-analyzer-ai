from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from resume_role_ai.models import JobPosition, Resume
from resume_role_ai.skill_matching import skill_terms


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_text(path: str | Path) -> str:
    """Extract readable text from a PDF, DOCX, or plain-text resume."""
    file_path = Path(path)
    extension = file_path.suffix.casefold()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported resume format: {extension or 'no extension'}")

    if extension == ".txt":
        text = file_path.read_text(encoding="utf-8")
    elif extension == ".docx":
        text = _extract_docx(file_path)
    else:
        text = _extract_pdf(file_path)

    normalized = re.sub(r"[ \t]+", " ", text)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not normalized:
        raise ValueError("No readable text was found in the resume")
    return normalized


def _extract_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            document = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as error:
        raise ValueError("The DOCX file is invalid or damaged") from error

    root = ElementTree.fromstring(document)
    paragraphs: list[str] = []
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    for paragraph in root.iter(f"{namespace}p"):
        line = "".join(node.text or "" for node in paragraph.iter(f"{namespace}t"))
        if line.strip():
            paragraphs.append(line.strip())
    return "\n".join(paragraphs)


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ValueError("PDF support requires the pypdf package") from error

    try:
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    except Exception as error:
        raise ValueError("The PDF file is invalid, damaged, or contains no extractable text") from error


def detect_skills(text: str, jobs: list[JobPosition]) -> list[str]:
    """Find known database skills in extracted text, preserving their canonical spelling."""
    detected: list[str] = []
    for skill in dict.fromkeys(skill for job in jobs for skill in job.required_skills):
        if any(
            re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, flags=re.IGNORECASE)
            for term in skill_terms(skill)
        ):
            detected.append(skill)
    return detected


def parse_resume(path: str | Path, resume_id: str, jobs: list[JobPosition]) -> Resume:
    text = extract_text(path)
    return Resume(id=resume_id, raw_text=text, skills=detect_skills(text, jobs))
