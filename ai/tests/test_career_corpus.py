import pandas as pd
import pytest

from resume_role_ai.career_corpus import _expected_job_indices, _ranking_metrics


def test_domain_rules_match_catalog_titles_without_per_resume_mapping() -> None:
    titles = [
        "Accountants and Auditors",
        "Financial and Investment Analysts",
        "Social Science Research Assistants",
        "Elementary School Teachers",
        "Tellers",
        "Fashion Designers",
    ]

    assert _expected_job_indices("ACCOUNTANT", titles) == {0}
    assert _expected_job_indices("Finance", titles) == {1}
    assert _expected_job_indices("Research Assistant", titles) == {2}
    assert _expected_job_indices("TEACHER", titles) == {3}
    assert _expected_job_indices("Banking", titles) == {4}
    assert _expected_job_indices("Apparel", titles) == {5}

    with pytest.raises(ValueError, match="Unsupported"):
        _expected_job_indices("Unknown", titles)


def test_calculates_ranking_metrics() -> None:
    metrics = _ranking_metrics(pd.DataFrame({"rank": [1, 5, 20]}))

    assert metrics["hit_at_1"] == 0.3333
    assert metrics["hit_at_5"] == 0.6667
    assert metrics["median_rank"] == 5
