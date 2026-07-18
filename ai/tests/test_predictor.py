import numpy as np

from resume_role_ai.models import JobPosition, Resume
from resume_role_ai.predictor import RolePredictor, load_jobs


def test_predicts_data_scientist_for_ml_resume() -> None:
    predictor = RolePredictor(load_jobs())
    resume = Resume(
        id="resume-1",
        raw_text="Built predictive machine learning models and analyzed datasets.",
        skills=["Python", "pandas", "scikit-learn", "SQL", "Machine Learning"],
    )

    result = predictor.predict(resume, limit=2)

    assert result.resume_id == "resume-1"
    assert result.matches[0].title == "Data Scientist"
    assert result.matches[0].score > result.matches[1].score
    assert "Python" in result.matches[0].matched_skills
    assert result.matches[0].missing_skills == []
    assert "Data Scientist" in result.matches[0].explanation


def test_limit_is_bounded_by_available_jobs() -> None:
    jobs = load_jobs()
    predictor = RolePredictor(jobs)
    resume = Resume(id="resume-2", raw_text="software development", skills=["C#"])

    assert len(predictor.predict(resume, limit=10_000).matches) == len(jobs)


def test_explains_missing_skills() -> None:
    predictor = RolePredictor(load_jobs())
    resume = Resume(id="resume-3", raw_text="Backend software developer", skills=["C#"])

    match = predictor.predict(resume, limit=1).matches[0]

    assert match.missing_skills
    assert "Strengthen the match" in match.explanation


def test_load_jobs_allows_an_empty_skill_list(tmp_path) -> None:
    catalog = tmp_path / "jobs.csv"
    catalog.write_text(
        "id,title,description,required_skills\nrole,Role,Role description,\n",
        encoding="utf-8",
    )

    assert load_jobs(catalog)[0].required_skills == []


def test_generic_skills_receive_less_weight_than_specific_skills() -> None:
    jobs = [
        JobPosition(
            id="onet-11-1000-00",
            title="Manager",
            description="Work",
            required_skills=["Writing", "Budgeting"],
        ),
        JobPosition(
            id="onet-13-1000-00",
            title="Analyst",
            description="Work",
            required_skills=["Writing", "Forecasting"],
        ),
        JobPosition(
            id="onet-15-1000-00",
            title="Developer",
            description="Work",
            required_skills=["Writing", "Kubernetes"],
        ),
    ]
    predictor = RolePredictor(jobs)

    assert predictor._skill_weights["kubernetes"] > predictor._skill_weights["writing"]


def test_family_filter_keeps_strong_family_and_enough_results() -> None:
    jobs = [
        JobPosition(id="onet-15-1000-00", title="Developer", description="Work", required_skills=[]),
        JobPosition(id="onet-15-1001-00", title="Tester", description="Work", required_skills=[]),
        JobPosition(
            id="onet-15-1002-00", title="Administrator", description="Work", required_skills=[]
        ),
        JobPosition(id="onet-11-1000-00", title="Manager", description="Work", required_skills=[]),
        JobPosition(id="onet-13-1000-00", title="Accountant", description="Work", required_skills=[]),
        JobPosition(id="onet-25-1000-00", title="Teacher", description="Work", required_skills=[]),
    ]
    predictor = RolePredictor(jobs)

    candidates = predictor._family_candidates(
        np.array([0.99, 0.8, 0.7, 0.6, 0.5, 0.4]), result_limit=5
    )

    assert {0, 1, 2}.issubset(set(candidates))
    assert len(candidates) >= 5


def test_common_skill_aliases_are_not_reported_missing() -> None:
    job = JobPosition(
        id="cloud-engineer",
        title="Cloud Engineer",
        description="Operate cloud services and delivery workflows",
        required_skills=[
            "Atlassian JIRA",
            "Amazon Web Services AWS software",
            "Microsoft Power BI",
        ],
    )
    predictor = RolePredictor([job])

    match = predictor.predict(
        Resume(
            id="alias-resume",
            raw_text="Used Jira, AWS, and Power BI",
            skills=["Jira", "AWS", "Power BI"],
        ),
        limit=1,
    ).matches[0]

    assert match.matched_skills == [
        "Atlassian JIRA",
        "Amazon Web Services AWS software",
        "Microsoft Power BI",
    ]
    assert match.missing_skills == []


def test_api_skill_satisfies_rest_api_requirement() -> None:
    job = JobPosition(
        id="backend-engineer",
        title="Backend Engineer",
        description="Build backend services",
        required_skills=["REST APIs"],
    )

    match = RolePredictor([job]).predict(
        Resume(id="api-resume", raw_text="Built APIs", skills=["APIs"]), limit=1
    ).matches[0]

    assert match.matched_skills == ["REST APIs"]
    assert match.missing_skills == []
