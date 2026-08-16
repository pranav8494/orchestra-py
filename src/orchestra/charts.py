"""The chart the model asks for, drawn twice: an HTML file, and text for the terminal.

Top level beside `artifacts.py`, not `core/`, because Plotly is a vendor library and §1.3
keeps `core/` free of them. Pure: no store, no file, no network — every function here
returns a string and the worker decides where it goes.

`ChartSpec` is a trust boundary (§7) but a deliberately permissive one. It is the shape
the model fills in, so it validates types and nothing else; whether the numbers can
actually be drawn is `insufficient_data`'s question, asked once and answered as data. A
validator would turn thin data into a failed run, and thin data is the case #7 has to
survive.
"""

import math
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# Two, because one point is a number, not a trend — there is nothing for a line to
# connect or a bar to be compared against.
MIN_POINTS = 2

# The worker uses a rejection verbatim as the subtask summary, so the opening is contract.
THIN_DATA_PREFIX = "Insufficient data to chart: "

_BAR = "█"

# All values equal leaves nothing to scale against, so every bar gets this instead.
_FLAT_BAR_LENGTH = 1

_VALUE_DECIMALS = 2

# Only a caller that skipped `insufficient_data` can reach this.
_NOTHING_TO_DRAW = "(no data to chart)"


class ChartKind(StrEnum):
    """The shapes we draw. A closed set, so an enum rather than bare strings (§7)."""

    BAR = "bar"
    LINE = "line"


class ChartSeries(BaseModel):
    """One line or one group of bars.

    Descriptions are prompt text: this model is published as part of the response schema
    (see `workers/visualization.py`), so the model reads them.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="What this series measures, in a few words.")
    values: list[float] = Field(
        default_factory=list,
        description="One number per category, in the same order as the categories.",
    )


class ChartSpec(BaseModel):
    """A chart, as the model describes it.

    Every field past the title defaults to empty. A reply that names only what it could
    see still parses, and `insufficient_data` reports it as thin rather than the provider
    reporting it as invalid.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="What the chart shows, as a short headline.")
    kind: ChartKind = Field(
        description="`line` for a series over time, `bar` for a comparison across categories."
    )
    x_label: str = Field(default="", description="Name of the horizontal axis.")
    y_label: str = Field(default="", description="Name of the vertical axis, with its units.")
    categories: list[str] = Field(
        default_factory=list,
        description="The points along the horizontal axis, in the order they should be drawn.",
    )
    series: list[ChartSeries] = Field(
        default_factory=list, description="The series to draw, each one value per category."
    )


def insufficient_data(spec: ChartSpec) -> str | None:
    """Say why `spec` cannot be drawn, or `None` when it can.

    The one place drawability is decided — the renderers below trust its verdict rather
    than each re-deriving it. The first problem is returned, not all of them: the message
    reaches the user as the subtask's summary, where one plain sentence beats a list.

    Ragged is judged strictly: one short series rejects the whole spec, rather than the
    good series being drawn alone. A chart missing a series it names misleads more quietly
    than no chart does.
    """
    if not spec.series:
        return f"{THIN_DATA_PREFIX}the model returned no series."
    if not spec.categories:
        return f"{THIN_DATA_PREFIX}the model returned no categories."
    if len(spec.categories) < MIN_POINTS:
        return (
            f"{THIN_DATA_PREFIX}{len(spec.categories)} point is not a trend, "
            f"{MIN_POINTS} are needed."
        )
    for series in spec.series:
        if len(series.values) != len(spec.categories):
            return (
                f"{THIN_DATA_PREFIX}series {series.name!r} has "
                f"{len(series.values)} values for {len(spec.categories)} categories."
            )
    # Shape is right, numbers are not: `nan`/`inf` plot as an empty figure, which reads as
    # a working chart. One finite value is enough — the renderers stub out the rest.
    if not any(math.isfinite(value) for series in spec.series for value in series.values):
        return f"{THIN_DATA_PREFIX}no value is a finite number."
    return None


def render_html(spec: ChartSpec) -> str:
    """Draw `spec` as a standalone HTML document.

    Assumes `insufficient_data(spec)` returned `None`; it is not re-checked here.

    Plotly's JS is linked from a CDN rather than inlined — it is ~3 MB, and the artifact
    store keeps every chart of every run. The file needs a network to open, which is the
    trade we take for not shipping the bundle a hundred times over.
    """
    # Deferred like the SDK import in `providers/base.py`: `agents/` imports this module
    # for `ChartSpec` alone, and only the run that draws should pay for Plotly.
    import plotly.graph_objects as go

    traces = [
        go.Bar(name=series.name, x=spec.categories, y=series.values)
        if spec.kind is ChartKind.BAR
        else go.Scatter(name=series.name, x=spec.categories, y=series.values, mode="lines+markers")
        for series in spec.series
    ]
    figure = go.Figure(data=traces)
    figure.update_layout(
        title=spec.title,
        xaxis_title=spec.x_label,
        yaxis_title=spec.y_label,
        template="plotly_white",
    )
    # `str()` because the SDK is untyped and §7 keeps `Any` from travelling out.
    return str(figure.to_html(full_html=True, include_plotlyjs="cdn"))


def render_ascii(spec: ChartSpec, *, width: int = 40) -> str:
    """Draw `spec` as plain text, for the terminal and for the report.

    Text, not Rich: this string is stored in an artifact the aggregator reads back, so it
    has to survive being a value in a JSON file (§5 — no styling outside `cli/`).

    Callers are expected to have run `insufficient_data` first. One that does not still
    gets a string: this never raises, on any spec.

    Args:
        width: the longest bar, in characters. No bar exceeds it.
    """
    lines = _heading(spec)
    # Truncated rather than rejected, for the same reason: degrade, don't raise.
    blocks = [
        [
            (label, value, _format_value(value))
            for label, value in zip(spec.categories, series.values, strict=False)
        ]
        for series in spec.series
    ]
    rows = [row for block in blocks for row in block]
    if not rows:
        return "\n".join([*lines, _NOTHING_TO_DRAW])

    # One scale across every series, so a bar means the same thing in every block.
    values = [value for _, value, _ in rows]
    baseline = min(0.0, min(values))
    span = max(values) - baseline
    label_width = max(len(label) for label, _, _ in rows)
    value_width = max(len(text) for _, _, text in rows)
    # Padded to the longest bar drawn, not to `width`: the value column still lines up,
    # without a screenful of trailing space when every bar is short.
    bar_width = max(_bar_length(value, baseline, span, width) for value in values)

    named = len(spec.series) > 1
    for series, block in zip(spec.series, blocks, strict=True):
        lines.append("")
        if named:
            lines.append(f"{series.name}:")
        lines += [
            f"  {label.ljust(label_width)}  "
            f"{(_BAR * _bar_length(value, baseline, span, width)).ljust(bar_width)}  "
            f"{text:>{value_width}}"
            for label, value, text in block
        ]
    return "\n".join(lines)


def _heading(spec: ChartSpec) -> list[str]:
    """The title and axis names, each skipped when the model left it blank."""
    lines = [spec.title] if spec.title else []
    axes = [
        f"{prefix}: {label}"
        for prefix, label in (("x", spec.x_label), ("y", spec.y_label))
        if label
    ]
    if axes:
        lines.append("    ".join(axes))
    return lines


def _bar_length(value: float, baseline: float, span: float, width: int) -> int:
    """Scale one value onto `0..width` characters.

    Measured from `baseline` (never above zero) rather than from the smallest value, so a
    negative number is a short bar instead of a negative-length one.
    """
    # A degenerate or non-finite span has no scale to divide by; a stub says "present"
    # without inventing a magnitude.
    if not math.isfinite(span) or span <= 0 or not math.isfinite(value):
        return _FLAT_BAR_LENGTH
    return max(0, min(width, round((value - baseline) / span * width)))


def _format_value(value: float) -> str:
    """Group thousands, drop a trailing `.0`, and never print a nonzero value as `0`."""
    if not math.isfinite(value):
        return str(value)  # `int()` raises on nan and inf, and the model can emit either
    if value.is_integer():
        return f"{int(value):,}"
    rounded = round(value, _VALUE_DECIMALS)
    return f"{rounded:,}" if rounded else f"{value:,}"
