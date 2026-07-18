import numpy as np

from resume_role_ai.matcher import SupervisedMatcher


def test_trained_matcher_scores_relevant_pair_higher(tmp_path) -> None:
    jobs = [
        "Python SQL data scientist machine learning",
        "Python SQL data scientist machine learning",
        "React CSS frontend developer",
        "React CSS frontend developer",
    ]
    resumes = [
        "Python SQL machine learning models",
        "React CSS user interfaces",
        "React CSS accessible interfaces",
        "Python SQL predictive models",
    ]
    labels = [1, 0, 1, 0]

    matcher = SupervisedMatcher.fit(jobs, resumes, labels)
    scores = matcher.predict_scores(
        ["Python SQL data scientist"] * 2,
        ["Python SQL model development", "React CSS interfaces"],
    )
    model_path = tmp_path / "matcher.joblib"
    matcher.save(model_path)
    loaded = SupervisedMatcher.load(model_path)

    assert scores[0] > scores[1]
    assert np.allclose(
        scores,
        loaded.predict_scores(
            ["Python SQL data scientist"] * 2,
            ["Python SQL model development", "React CSS interfaces"],
        ),
    )
