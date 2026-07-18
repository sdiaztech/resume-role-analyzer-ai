import zipfile

import pytest

from resume_role_ai.models import JobPosition
from resume_role_ai.parser import detect_skills, extract_text, parse_resume
from resume_role_ai.predictor import load_jobs


def test_parses_text_and_detects_known_skills(tmp_path) -> None:
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text("Data scientist experienced with Python, SQL and scikit-learn.")

    resume = parse_resume(resume_file, "text-resume", load_jobs())

    assert resume.id == "text-resume"
    assert {"Python", "SQL", "scikit-learn"}.issubset(resume.skills)


def test_extracts_docx_without_requiring_office(tmp_path) -> None:
    resume_file = tmp_path / "resume.docx"
    document_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body><w:p><w:r><w:t>Backend Engineer</w:t></w:r></w:p>
      <w:p><w:r><w:t>Python and C#</w:t></w:r></w:p></w:body>
    </w:document>"""
    with zipfile.ZipFile(resume_file, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    assert extract_text(resume_file) == "Backend Engineer\nPython and C#"


def test_rejects_unsupported_files(tmp_path) -> None:
    resume_file = tmp_path / "resume.rtf"
    resume_file.write_text("resume")

    with pytest.raises(ValueError, match="Unsupported resume format"):
        extract_text(resume_file)


def test_skill_detection_does_not_match_substrings() -> None:
    skills = detect_skills("I use JavaScript, not Java.", load_jobs())

    assert "JavaScript" in skills
    assert "Java" not in skills


def test_skill_detection_recognizes_common_vendor_aliases() -> None:
    jobs = [
        JobPosition(
            id="cloud-role",
            title="Cloud Engineer",
            description="Cloud infrastructure work",
            required_skills=[
                "Atlassian JIRA",
                "Amazon Web Services AWS software",
                "Microsoft Power BI",
            ],
        )
    ]

    text = "Managed Jira tickets, deployed services to AWS, and built Power BI reports."
    assert detect_skills(text, jobs) == [
        "Atlassian JIRA",
        "Amazon Web Services AWS software",
        "Microsoft Power BI",
    ]


def test_platform_acronym_does_not_imply_specific_vendor_products() -> None:
    jobs = [
        JobPosition(
            id="cloud-role",
            title="Cloud Engineer",
            description="Cloud infrastructure work",
            required_skills=[
                "Amazon Web Services AWS software",
                "Amazon Web Services AWS CloudFormation",
            ],
        )
    ]

    assert detect_skills("Deployed applications to AWS.", jobs) == [
        "Amazon Web Services AWS software"
    ]
    assert detect_skills("Provisioned infrastructure with AWS CloudFormation.", jobs) == [
        "Amazon Web Services AWS software",
        "Amazon Web Services AWS CloudFormation",
    ]


def test_api_alias_matches_rest_api_catalog_skill() -> None:
    jobs = [
        JobPosition(
            id="software-role",
            title="Software Engineer",
            description="Build backend services",
            required_skills=["REST APIs"],
        )
    ]

    assert detect_skills("Designed and maintained APIs for customer applications.", jobs) == [
        "REST APIs"
    ]
