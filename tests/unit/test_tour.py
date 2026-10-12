"""Guided tour contract: ordered steps, every step has a target page and text."""

from app.components.tour import TOUR_STEPS, clamp_step, get_tour_step, tour_step_count


def test_tour_has_covering_steps():
    assert tour_step_count() >= 5
    ids = [step["id"] for step in TOUR_STEPS]
    assert ids[0] == "overview"
    assert ids[-1] == "evaluations"
    assert len(set(ids)) == len(ids)


def test_every_step_has_page_title_body():
    for step in TOUR_STEPS:
        assert step["page"].endswith(".py")
        assert len(step["title"]) > 0
        assert len(step["body"]) > 10


def test_step_navigation_clamps():
    assert clamp_step(-5) == 0
    assert clamp_step(10**6) == tour_step_count() - 1
    assert get_tour_step(0)["id"] == "overview"
