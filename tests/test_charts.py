"""Tests for the chart renderers.

Strings only: nothing here writes a file or opens a socket, and Plotly's markup is the
library's business, so `render_html` is asserted on the few substrings we put there (§12).
"""

from itertools import takewhile

from orchestra.charts import (
    MIN_POINTS,
    THIN_DATA_PREFIX,
    ChartKind,
    ChartSeries,
    ChartSpec,
    insufficient_data,
    render_ascii,
    render_html,
)

NAN = float("nan")
INF = float("inf")


def bar_spec(**overrides: object) -> ChartSpec:
    """A drawable bar chart. Overrides let one field be broken at a time."""
    fields: dict[str, object] = {
        "title": "Revenue by quarter",
        "kind": ChartKind.BAR,
        "x_label": "Quarter",
        "y_label": "USD (millions)",
        "categories": ["Q1", "Q2", "Q3"],
        "series": [ChartSeries(name="Revenue", values=[1200.0, 1310.5, 980.0])],
    }
    return ChartSpec.model_validate(fields | overrides)


def line_spec() -> ChartSpec:
    return ChartSpec(
        title="Signups over time",
        kind=ChartKind.LINE,
        x_label="Month",
        y_label="Signups",
        categories=["Jan", "Feb", "Mar"],
        series=[ChartSeries(name="Signups", values=[10.0, 24.0, 31.0])],
    )


# --- the spec is permissive on purpose -------------------------------------------------


def test_chart_spec_sparse_reply_still_parses() -> None:
    """A model that names only a title and a kind must not fail validation — thin data is
    reported by `insufficient_data`, never raised at the trust boundary."""
    spec = ChartSpec(title="Nothing yet", kind=ChartKind.BAR)

    assert spec.categories == []
    assert spec.series == []
    assert spec.x_label == spec.y_label == ""


def test_chart_spec_ragged_series_still_parses() -> None:
    spec = bar_spec(series=[ChartSeries(name="Revenue", values=[1.0])])

    assert spec.series[0].values == [1.0]


# --- insufficient_data ----------------------------------------------------------------


def test_insufficient_data_good_bar_spec_returns_none() -> None:
    assert insufficient_data(bar_spec()) is None


def test_insufficient_data_good_line_spec_returns_none() -> None:
    assert insufficient_data(line_spec()) is None


def test_insufficient_data_no_series_returns_message() -> None:
    reason = insufficient_data(bar_spec(series=[]))

    assert reason is not None
    assert reason.startswith(THIN_DATA_PREFIX)


def test_insufficient_data_no_categories_returns_message() -> None:
    reason = insufficient_data(bar_spec(categories=[], series=[ChartSeries(name="Revenue")]))

    assert reason is not None
    assert reason.startswith(THIN_DATA_PREFIX)


def test_insufficient_data_single_category_returns_message() -> None:
    reason = insufficient_data(
        bar_spec(categories=["Q1"], series=[ChartSeries(name="Revenue", values=[1200.0])])
    )

    assert reason is not None
    assert reason.startswith(THIN_DATA_PREFIX)
    assert str(MIN_POINTS) in reason


def test_insufficient_data_empty_series_is_rejected_as_ragged() -> None:
    """An empty series is a ragged one by then — no categories' worth of values."""
    reason = insufficient_data(
        bar_spec(series=[ChartSeries(name="Revenue"), ChartSeries(name="Cost")])
    )

    assert reason is not None
    assert reason.startswith(THIN_DATA_PREFIX)
    assert "Revenue" in reason


def test_insufficient_data_ragged_series_names_the_offender() -> None:
    reason = insufficient_data(
        bar_spec(
            series=[
                ChartSeries(name="Revenue", values=[1.0, 2.0, 3.0]),
                ChartSeries(name="Cost", values=[1.0]),
            ]
        )
    )

    assert reason is not None
    assert reason.startswith(THIN_DATA_PREFIX)
    assert "Cost" in reason
    assert "Revenue" not in reason  # the first problem, not a list of them


def test_insufficient_data_all_values_non_finite_returns_message() -> None:
    """Regression: a well-shaped spec of `nan`/`inf` used to pass, and the worker then
    wrote a real Plotly file plotting nothing."""
    reason = insufficient_data(
        bar_spec(categories=["Q1", "Q2"], series=[ChartSeries(name="Rev", values=[NAN, INF])])
    )

    assert reason is not None
    assert reason.startswith(THIN_DATA_PREFIX)


def test_insufficient_data_one_finite_value_is_enough_to_draw() -> None:
    reason = insufficient_data(bar_spec(series=[ChartSeries(name="Rev", values=[NAN, 2.0, INF])]))

    assert reason is None


def test_insufficient_data_every_rejection_starts_with_the_phrase() -> None:
    """The worker uses the message verbatim as the subtask summary, so the opening is
    part of the contract."""
    broken = [
        bar_spec(series=[]),
        bar_spec(categories=[], series=[ChartSeries(name="Revenue")]),
        bar_spec(categories=["Q1"], series=[ChartSeries(name="Revenue", values=[1.0])]),
        bar_spec(series=[ChartSeries(name="Revenue")]),
        bar_spec(series=[ChartSeries(name="Revenue", values=[1.0])]),
        bar_spec(series=[ChartSeries(name="Revenue", values=[NAN, NAN, INF])]),
    ]

    reasons = [insufficient_data(spec) for spec in broken]

    assert all(reason is not None and reason.startswith(THIN_DATA_PREFIX) for reason in reasons)


# --- render_ascii ---------------------------------------------------------------------


def test_render_ascii_bar_spec_contains_labels_and_values() -> None:
    text = render_ascii(bar_spec())

    assert "Revenue by quarter" in text
    for category in ("Q1", "Q2", "Q3"):
        assert category in text
    assert "1,200" in text  # grouped, and no trailing `.0`
    assert "1,310.5" in text
    assert "1200.0" not in text


def test_render_ascii_line_spec_renders() -> None:
    text = render_ascii(line_spec())

    assert "Signups over time" in text
    assert "Jan" in text
    assert "31" in text


def test_render_ascii_axis_labels_appear_when_set() -> None:
    text = render_ascii(bar_spec())

    assert "Quarter" in text
    assert "USD (millions)" in text


def test_render_ascii_blank_axis_labels_are_omitted() -> None:
    text = render_ascii(bar_spec(x_label="", y_label=""))

    assert "x:" not in text
    assert "y:" not in text


def test_render_ascii_multi_series_contains_both_names() -> None:
    text = render_ascii(
        bar_spec(
            series=[
                ChartSeries(name="Revenue", values=[1.0, 2.0, 3.0]),
                ChartSeries(name="Cost", values=[0.5, 1.5, 2.5]),
            ]
        )
    )

    assert "Revenue" in text
    assert "Cost" in text


def test_render_ascii_uses_one_scale_across_every_series() -> None:
    """A bar means the same quantity in every block. Scaled per series instead, the small
    series would stretch to the same lengths as the large one."""
    text = render_ascii(
        bar_spec(
            series=[
                ChartSeries(name="Small", values=[1.0, 2.0, 3.0]),
                ChartSeries(name="Large", values=[10.0, 20.0, 30.0]),
            ]
        )
    )

    assert max(_bars(text, "Small")) < min(_bars(text, "Large"))


def _bars(text: str, series_name: str) -> list[int]:
    """Bar characters per row of one series' block, which runs to the next blank line."""
    lines = text.splitlines()
    rows = lines[lines.index(f"{series_name}:") + 1 :]
    return [row.count("█") for row in takewhile(bool, rows)]


def test_render_ascii_non_finite_value_is_printed_as_is() -> None:
    """A stray `nan` beside real numbers still draws; `int()` would raise on it."""
    text = render_ascii(bar_spec(series=[ChartSeries(name="Mixed", values=[1.0, NAN, 3.0])]))

    assert "nan" in text


def test_render_ascii_single_series_omits_the_series_heading() -> None:
    """One series needs no legend; the title already says what is drawn."""
    text = render_ascii(bar_spec())

    assert "Revenue:" not in text


def test_render_ascii_all_equal_values_does_not_raise() -> None:
    """Regression: a zero span is a division by zero if bars are scaled naively."""
    text = render_ascii(bar_spec(series=[ChartSeries(name="Flat", values=[0.0, 0.0, 0.0])]))

    assert "Q1" in text


def test_render_ascii_negative_values_produce_bars() -> None:
    text = render_ascii(bar_spec(series=[ChartSeries(name="Delta", values=[-5.0, 0.0, 10.0])]))

    assert "█" in text
    assert "-5" in text


def test_render_ascii_longest_bar_respects_width() -> None:
    width = 12
    text = render_ascii(bar_spec(), width=width)

    assert max(line.count("█") for line in text.splitlines()) <= width


def test_render_ascii_rejected_spec_does_not_raise() -> None:
    """The worker only calls this after the check passes, but the degraded path must not
    be able to fail a second time."""
    assert render_ascii(ChartSpec(title="Empty", kind=ChartKind.BAR))
    assert render_ascii(bar_spec(series=[ChartSeries(name="Ragged", values=[1.0])]))


def test_render_ascii_returns_plain_text() -> None:
    assert "\x1b" not in render_ascii(bar_spec())


# --- render_html ----------------------------------------------------------------------


def test_render_html_bar_spec_contains_document_title_and_series() -> None:
    html = render_html(bar_spec())

    assert "<html" in html
    assert "Revenue by quarter" in html
    assert "Revenue" in html


def test_render_html_line_spec_renders() -> None:
    html = render_html(line_spec())

    assert "<html" in html
    assert "Signups over time" in html


def test_render_html_multi_series_names_every_series() -> None:
    html = render_html(
        bar_spec(
            series=[
                ChartSeries(name="Revenue", values=[1.0, 2.0, 3.0]),
                ChartSeries(name="Cost", values=[0.5, 1.5, 2.5]),
            ]
        )
    )

    assert "Revenue" in html
    assert "Cost" in html
