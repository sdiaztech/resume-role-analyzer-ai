import pandas as pd

from resume_role_ai.dataset import prepare_training_pairs


def test_prepares_balanced_deterministic_pairs(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        [
            {
                "resume_id": "R1",
                "summary": "Python developer",
                "skills": ["Python"],
                "experience_bullets": ["Built APIs"],
            },
            {
                "resume_id": "R2",
                "summary": "Product manager",
                "skills": ["Roadmaps"],
                "experience_bullets": ["Led discovery"],
            },
        ]
    ).to_parquet(source / "resumes.parquet", index=False)
    pd.DataFrame(
        [
            {
                "job_id": "J1",
                "job_title": "Developer",
                "description": "Build software",
                "must_have_skills": ["Python"],
                "requirements": ["API experience"],
            }
        ]
    ).to_parquet(source / "jobs.parquet", index=False)
    pd.DataFrame([{"job_id": "J1", "relevant_resume_ids": ["R1"]}]).to_parquet(
        source / "matches.parquet", index=False
    )

    first = prepare_training_pairs(source, tmp_path / "first.parquet")
    second = prepare_training_pairs(source, tmp_path / "second.parquet")

    assert first.equals(second)
    assert first["label"].value_counts().to_dict() == {1: 1, 0: 1}
    assert set(first["resume_id"]) == {"R1", "R2"}
    assert (tmp_path / "first.parquet").exists()
