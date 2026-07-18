import numpy as np

from resume_role_ai.semantic import (
    SentenceEmbeddingRanker,
    rank_percentiles,
)


class FakeEncoder:
    def encode(self, texts, **_kwargs):
        return np.array(
            [
                [
                    float("bank" in text.casefold()),
                    float("fashion" in text.casefold()),
                    float("teach" in text.casefold()),
                ]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_rank_percentiles_preserve_order() -> None:
    ranked = rank_percentiles(np.array([0.2, 0.9, 0.5]))

    assert ranked.tolist() == [0.0, 1.0, 0.5]


def test_sentence_embedding_ranker_prefers_semantically_related_text() -> None:
    ranker = SentenceEmbeddingRanker(
        ["Bank accounts", "Fashion design", "Teaching students"], encoder=FakeEncoder()
    )

    scores = ranker.predict_scores("Worked at a bank")

    assert int(scores.argmax()) == 0
