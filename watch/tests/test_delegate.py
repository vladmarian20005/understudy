"""Delegate: where the object goes."""
from watch import delegate

SCORE = {"giver": "left", "taker": "right", "target_cm": None}


def test_the_target_is_asked_for_else_the_scores_else_in_front_of_the_taker():
    assert delegate.target_for(SCORE) == [12.0, 24.0]
    assert delegate.target_for(dict(SCORE, taker="left")) == [-12.0, 24.0]
    assert delegate.target_for(dict(SCORE, target_cm=[10, 20])) == [10.0, 20.0]
    assert delegate.target_for(dict(SCORE, target_cm=[10, 20]), override=[-5, 30]) == [-5.0, 30.0]
