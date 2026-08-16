"""Tests for the repetition detector (#9).

The contract is what the signature counts as the same step, and when a run of them
trips — including that the window slides, so an old streak stops mattering.
"""

import pytest

from orchestra.core.loop import DEFAULT_MAX_REPEATS, RepetitionDetector, step_signature


def _signature(
    name: str = "fetch_data",
    arguments: dict[str, object] | None = None,
    output: str = "42 rows",
) -> str:
    return step_signature(name, {"query": "revenue"} if arguments is None else arguments, output)


def test_step_signature_identical_inputs_match() -> None:
    assert _signature() == _signature()


def test_step_signature_different_tool_name_differs() -> None:
    assert _signature(name="run_python") != _signature()


def test_step_signature_different_argument_value_differs() -> None:
    assert _signature(arguments={"query": "costs"}) != _signature()


def test_step_signature_different_output_differs() -> None:
    """Same call, new result, is progress — not a repeat."""
    assert _signature(output="0 rows") != _signature()


def test_step_signature_reordered_argument_keys_match() -> None:
    left = _signature(arguments={"query": "revenue", "limit": 10})
    right = _signature(arguments={"limit": 10, "query": "revenue"})

    assert left == right


def test_step_signature_shifted_part_boundary_differs() -> None:
    """The parts are separated, so ("a", "b") cannot collide with ("ab", "")."""
    assert step_signature("a", {}, "b") != step_signature("ab", {}, "")


def test_record_alternating_signatures_never_trips() -> None:
    detector = RepetitionDetector()

    assert not any(detector.record(_signature(output=str(step))) for step in range(50))


def test_record_repeat_within_limit_does_not_trip() -> None:
    detector = RepetitionDetector()
    signature = _signature()

    assert not any(detector.record(signature) for _ in range(DEFAULT_MAX_REPEATS))


def test_record_repeat_over_limit_trips() -> None:
    detector = RepetitionDetector()
    signature = _signature()
    for _ in range(DEFAULT_MAX_REPEATS):
        detector.record(signature)

    assert detector.record(signature)


def test_record_repeats_aged_out_of_the_window_do_not_trip() -> None:
    """Five repeats, a full window of other work, then one more: the streak is gone."""
    detector = RepetitionDetector()
    signature = _signature()
    for _ in range(DEFAULT_MAX_REPEATS):
        detector.record(signature)
    for step in range(detector.window):
        detector.record(_signature(output=str(step)))

    assert not detector.record(signature)


def test_record_keeps_working_after_tripping() -> None:
    """The caller may ignore a trip; the detector must still trip on the next repeat."""
    detector = RepetitionDetector()
    signature = _signature()
    for _ in range(DEFAULT_MAX_REPEATS + 1):
        detector.record(signature)

    assert detector.record(signature)


@pytest.mark.parametrize("bound", ["window", "max_repeats"])
def test_init_non_positive_bound_raises(bound: str) -> None:
    with pytest.raises(ValueError, match=f"{bound} must be at least 1"):
        RepetitionDetector(**{bound: 0})
