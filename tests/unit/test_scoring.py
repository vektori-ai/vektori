"""Unit tests for fact scoring."""

from datetime import datetime, timedelta

from vektori.retrieval.scoring import score_and_rank


def _fact(distance=0.1, confidence=1.0, days_ago=0):
    return {
        "id": "fact-test",
        "text": "test fact",
        "distance": distance,
        "confidence": confidence,
        "created_at": datetime.utcnow() - timedelta(days=days_ago),
    }


def test_lower_distance_higher_score():
    facts = [_fact(distance=0.3), _fact(distance=0.1)]
    scored = score_and_rank(facts)
    assert scored[0]["distance"] == 0.1  # closer = first


def test_score_field_added():
    scored = score_and_rank([_fact()])
    assert "score" in scored[0]
    assert 0 <= scored[0]["score"] <= 1.0


def test_temporal_decay():
    recent = _fact(distance=0.1, days_ago=0)
    old = _fact(distance=0.1, days_ago=365)
    scored = score_and_rank([recent, old], temporal_decay_rate=0.01)
    # Same distance and confidence — recency should differ
    assert scored[0]["score"] > scored[1]["score"]


def test_confidence_affects_score():
    high = _fact(distance=0.1, confidence=1.0)
    low = _fact(distance=0.1, confidence=0.5)
    scored = score_and_rank([high, low])
    assert scored[0]["confidence"] == 1.0
    assert scored[0]["score"] > scored[1]["score"]


def test_empty_list():
    assert score_and_rank([]) == []


def test_sorted_descending():
    facts = [_fact(distance=0.5), _fact(distance=0.1), _fact(distance=0.3)]
    scored = score_and_rank(facts)
    scores = [f["score"] for f in scored]
    assert scores == sorted(scores, reverse=True)
