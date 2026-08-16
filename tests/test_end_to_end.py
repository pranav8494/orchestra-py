"""The QA scenarios, driven end to end through `run_once` and the CLI over `FakeProvider` (#16).

Every other module tests one part against doubles. This one substitutes the provider port
and nothing else: the real planner, the real workers, the real tools over the bundled
`data/`, the real aggregator, and — for the CLI cases — the real Typer command. What is
scripted is what a model would say; everything under it runs.

The scripts live here rather than in `tests/scenarios.py`, whose contract is the *planner's*
shape: these are the *workers'* conversations, and they change whenever a tool's schema does.

`FAN_OUT` is the reason `FakeProvider.turns_by_topic` exists — its two retrievals are
dispatched together and interleave on `asyncio.to_thread`, so one flat queue would hand one
branch the other's turn. Routing is on the subtask's own instruction, which
`build_briefing` puts in the first user turn.
"""

import asyncio
import json
from typing import cast

import pytest
from typer.testing import CliRunner

import orchestra.cli.app as cli_app
from conftest import (
    FakeProvider,
    OfflineRun,
    ScriptedAsker,
    ScriptedChat,
    answer_turn,
    tool_turn,
)
from orchestra.agents import toolsets
from orchestra.agents.aggregator import FigureDraft, ReportDraft
from orchestra.agents.interrupt import InterruptAction, InterruptDraft
from orchestra.agents.planner import PlannerAction, PlannerDraft, SubtaskDraft
from orchestra.agents.structured import DEFAULT_MAX_RETRIES
from orchestra.agents.toolsets import QUERY_CSV_TOOL, SEARCH_CORPUS, SEARCH_TOOL
from orchestra.agents.workers.visualization import ChartDraft
from orchestra.app import run_once
from orchestra.artifacts import ArtifactStore
from orchestra.charts import ChartKind, ChartSeries, ChartSpec
from orchestra.cli.app import app
from orchestra.cli.console import err_console
from orchestra.config import default_data_dir
from orchestra.core.errors import ExitCode
from orchestra.core.question import Question, QuestionKind
from orchestra.core.state import (
    ARTIFACT_PREFIX,
    AgentRole,
    EventKind,
    SubtaskStatus,
    artifact_path,
)
from orchestra.providers.base import AssistantTurn
from orchestra.tools.python_exec import TOOL_NAME as RUN_PYTHON_TOOL
from orchestra.tools.python_exec import RunPythonTool
from scenarios import (
    FAN_OUT,
    LINEAR,
    ROLE_OMISSION,
    SCENARIOS,
    Scenario,
    assert_plan_shape,
    scenario_id,
)

runner = CliRunner()

SUMMARY = "Revenue grew in each of the last three quarters."

# How many rows every scripted `query_csv` call asks for, and so what the analysis scripts
# must count.
QUARTERS = 3

# 1 + the retries `agents/structured.py` allows: how many drafts one structured call spends
# before it gives up. Derived, so raising the ceiling does not silently shorten a script.
STRUCTURED_ATTEMPTS = 1 + DEFAULT_MAX_RETRIES

# Long enough for a subprocess to start on a loaded CI box, short enough that a run which
# stopped making progress fails instead of stalling the suite (AC3: never a hang).
BOUND_SECONDS = 60.0

# Fast enough to keep the timeout test cheap, slow enough that the child really starts.
EXECUTOR_TIMEOUT = 0.2


def _step(scenario: Scenario, role: AgentRole, index: int = 0) -> SubtaskDraft:
    """One of a scenario's drafted subtasks, so ids and instructions here track the draft
    rather than repeating it — a renamed step fails on the name, not on a stale literal."""
    return [subtask for subtask in scenario.draft().subtasks if subtask.role is role][index]


LINEAR_FETCH = _step(LINEAR, AgentRole.DATA_RETRIEVAL)
LINEAR_ANALYSIS = _step(LINEAR, AgentRole.ANALYTICS)
LINEAR_CHART = _step(LINEAR, AgentRole.VISUALIZATION)

FAN_OUT_CSV = _step(FAN_OUT, AgentRole.DATA_RETRIEVAL, 0)
FAN_OUT_SEARCH = _step(FAN_OUT, AgentRole.DATA_RETRIEVAL, 1)
FAN_OUT_COMPARE = _step(FAN_OUT, AgentRole.ANALYTICS)

OMISSION_FETCH = _step(ROLE_OMISSION, AgentRole.DATA_RETRIEVAL)
OMISSION_ANALYSIS = _step(ROLE_OMISSION, AgentRole.ANALYTICS)

# The id a mid-run replan gives the step it puts in place of the chart (#12). A new one:
# a replacement may consume a finished step but may not reuse a live step's id.
REPLANNED_STEP = "summarise_growth"

# Whole lowercase tokens the bundled corpus answers to (`saas`, `growth`, `benchmarks`,
# `industry`, `peer`). A query sharing none of them matches nothing, the loop drops the
# empty result, and the branch fails with "finished without retrieving anything".
SEARCH_QUERY = "industry saas growth benchmarks peer"

# One category of the scripted chart, so an assertion on what the terminal drew tracks the
# fixture rather than a literal.
CHART_CATEGORY = "2025Q3"


def _file(step: SubtaskDraft) -> str:
    """The artifact a worker writes for `step` — its id, as the store names it."""
    return f"{step.id}.json"


def _pointer(step: SubtaskDraft) -> str:
    """The pointer that artifact is registered under."""
    return f"{ARTIFACT_PREFIX}{_file(step)}"


# Real scripts in a real subprocess: this is what proves one worker's pointer resolves
# inside the next one's executor. stdlib only — a pandas import would cost a second per
# case and prove nothing more.
_COUNT_ROWS = """\
import json
data = json.load(open("{dataset}"))
rows = [row for table in data["datasets"] for row in table["csv"].splitlines()[1:] if row]
print("quarters analysed:", len(rows))
"""

_COMPARE = """\
import json
ours = json.load(open("{ours}"))
theirs = json.load(open("{theirs}"))
rows = [row for table in ours["datasets"] for row in table["csv"].splitlines()[1:] if row]
print("quarters analysed:", len(rows), "benchmark notes:", len(theirs["sources"]))
"""


def _csv_turns() -> list[AssistantTurn | BaseException]:
    """A retrieval agent reading the company's own figures, then closing."""
    return [
        tool_turn(QUERY_CSV_TOOL, last_n=QUARTERS),
        answer_turn("Retrieved the last three quarters."),
    ]


def _search_turns() -> list[AssistantTurn | BaseException]:
    """A retrieval agent reading background, then closing."""
    return [
        tool_turn(SEARCH_TOOL, query=SEARCH_QUERY, limit=3),
        answer_turn("Found the peer-group growth benchmarks."),
    ]


def _python_turns(code: str, inputs: list[str]) -> list[AssistantTurn | BaseException]:
    """An analytics agent running one script over `inputs`, then closing.

    `inputs[0]` becomes the figure's source, so the pointer order is load-bearing.
    """
    return [
        tool_turn(RUN_PYTHON_TOOL, code=code, inputs=inputs),
        answer_turn("Growth was positive in every quarter."),
    ]


def _chart_draft() -> ChartDraft:
    """A drawable chart — two points, so the step completes rather than degrading."""
    return ChartDraft(
        summary="Revenue rose in each quarter.",
        spec=ChartSpec(
            title="Quarterly revenue",
            kind=ChartKind.LINE,
            x_label="Quarter",
            y_label="Revenue",
            categories=[CHART_CATEGORY, "2025Q4"],
            series=[ChartSeries(name="Revenue", values=[6_340_000.0, 7_015_000.0])],
        ),
    )


def _report_draft(source: str) -> ReportDraft:
    """A report citing one figure. `source` must be a pointer the run really minted, or
    `Aggregator.validate` rejects the draft and the report degrades."""
    return ReportDraft(
        executive_summary=SUMMARY,
        key_figures=[FigureDraft(label="Quarters analysed", value=str(QUARTERS), source=source)],
    )


def _linear_provider() -> FakeProvider:
    """Fetch, analyse, chart — one branch at a time, so one flat queue serves it."""
    return FakeProvider(
        responses=[LINEAR.draft(), _chart_draft(), _report_draft(_pointer(LINEAR_ANALYSIS))],
        turns=[
            *_csv_turns(),
            *_python_turns(
                _COUNT_ROWS.format(dataset=_file(LINEAR_FETCH)), [_pointer(LINEAR_FETCH)]
            ),
        ],
    )


def _fan_out_provider() -> FakeProvider:
    """Two retrievals at once, then the comparison and the chart.

    One queue per branch: the retrievals run concurrently, so which turn answers which
    `send` cannot be decided by arrival order. Retrieval and analytics make no structured
    call, so `responses` still holds only the plan, the chart and the report.
    """
    return FakeProvider(
        responses=[FAN_OUT.draft(), _chart_draft(), _report_draft(_pointer(FAN_OUT_COMPARE))],
        turns_by_topic={
            FAN_OUT_CSV.instruction: _csv_turns(),
            FAN_OUT_SEARCH.instruction: _search_turns(),
            FAN_OUT_COMPARE.instruction: _python_turns(
                _COMPARE.format(ours=_file(FAN_OUT_CSV), theirs=_file(FAN_OUT_SEARCH)),
                [_pointer(FAN_OUT_CSV), _pointer(FAN_OUT_SEARCH)],
            ),
        },
    )


def _role_omission_provider() -> FakeProvider:
    """Fetch and summarise. No `ChartDraft` at all — the run must never ask for one."""
    return FakeProvider(
        responses=[ROLE_OMISSION.draft(), _report_draft(_pointer(OMISSION_ANALYSIS))],
        turns=[
            *_csv_turns(),
            *_python_turns(
                _COUNT_ROWS.format(dataset=_file(OMISSION_FETCH)), [_pointer(OMISSION_FETCH)]
            ),
        ],
    )


_PROVIDERS = {
    LINEAR.name: _linear_provider,
    FAN_OUT.name: _fan_out_provider,
    ROLE_OMISSION.name: _role_omission_provider,
}


def _provider(scenario: Scenario) -> FakeProvider:
    """The model's whole side of one scenario, fresh — a shared one is one a test leaves
    half-drained."""
    return _PROVIDERS[scenario.name]()


def _corpus_snippet(keyword: str) -> str:
    """The bundled note answering `keyword`, read from the file the run itself searches.

    Read rather than quoted: an assertion on a copy of the corpus passes the day the corpus
    changes and the run stops finding anything.
    """
    entries = json.loads((default_data_dir() / SEARCH_CORPUS).read_text(encoding="utf-8"))
    return cast("str", next(entry["snippet"] for entry in entries if keyword in entry["keywords"]))


# --------------------------------------------------------------------------
# Criterion 1 — all three scenarios, end to end, against the bundled dataset.
# The next two sections carry the halves of it the shape alone cannot state.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=scenario_id)
@pytest.mark.asyncio
async def test_scenario_runs_every_step_and_leaves_every_pointer_resolvable(
    scenario: Scenario, offline_run: OfflineRun
) -> None:
    """Each scenario plans its required shape, runs every subtask to done through the real
    workers, and hands back a report whose artifacts are all on disk."""
    provider = _provider(scenario)
    offline_run(provider)

    async with asyncio.timeout(BOUND_SECONDS):
        state = await run_once(scenario.prompt)

    assert state.plan is not None
    assert_plan_shape(state.plan, scenario.shape)  # the planner's contract still holds (#17)
    assert [subtask.status for subtask in state.plan.subtasks] == [SubtaskStatus.DONE] * len(
        state.plan.subtasks
    )
    assert not state.failed
    assert state.artifact_dir is not None
    assert set(state.artifacts) == {subtask.id for subtask in state.plan.subtasks}
    for pointer in state.artifacts.values():
        assert artifact_path(state.artifact_dir, pointer).is_file()
    report = state.final_result
    assert report is not None
    assert report.executive_summary == SUMMARY
    # The figure survived the ledger's backing filter, so it cites this run's own artifact.
    assert [figure.source for figure in report.key_figures] == [
        state.artifacts[_step(scenario, AgentRole.ANALYTICS).id]
    ]


# --------------------------------------------------------------------------
# Criterion 1, continued — the fan-out actually fans out.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_starts_both_retrievals_before_either_completes(
    offline_run: OfflineRun,
) -> None:
    """The two independent branches overlap, and both reach the step that consumes them:
    measured on the event stream and on the analysis output, never on the wall clock."""
    provider = _provider(FAN_OUT)
    offline_run(provider)

    async with asyncio.timeout(BOUND_SECONDS):
        state = await run_once(FAN_OUT.prompt)

    first_completion = next(
        index
        for index, event in enumerate(state.events)
        if event.kind is EventKind.SUBTASK_COMPLETED
    )
    running = {
        event.subtask_id
        for event in state.events[:first_completion]
        if event.kind is EventKind.SUBTASK_STARTED
    }
    assert running == {FAN_OUT_CSV.id, FAN_OUT_SEARCH.id}

    assert state.artifact_dir is not None
    store = ArtifactStore(state.artifact_dir)
    # The search branch answered from the bundled corpus, not from an empty result the
    # loop would have dropped.
    retrieved = json.loads(store.get_text(state.artifacts[FAN_OUT_SEARCH.id]))
    assert _corpus_snippet("benchmarks") in retrieved["sources"][0]["result"]
    # And both branches' artifacts were staged in the analytics subprocess, which is what
    # makes this a fan-in rather than two runs.
    analysis = json.loads(store.get_text(state.artifacts[FAN_OUT_COMPARE.id]))
    assert analysis["figures"][0]["value"].strip() == "quarters analysed: 3 benchmark notes: 1"


# --------------------------------------------------------------------------
# Criterion 1, continued — the omitted role stays omitted.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_role_omission_reports_no_chart_and_writes_no_html(
    offline_run: OfflineRun,
) -> None:
    """A plan with no visualization step produces no chart on the ledger and none on disk.

    The drained queues are what make "no `ChartDraft` was needed" a claim about the run
    rather than about an unused queue entry.
    """
    provider = _provider(ROLE_OMISSION)
    offline_run(provider)

    async with asyncio.timeout(BOUND_SECONDS):
        state = await run_once(ROLE_OMISSION.prompt)

    report = state.final_result
    assert report is not None
    assert report.chart is None
    assert report.chart_ascii is None
    assert state.artifact_dir is not None
    assert [path.name for path in state.artifact_dir.iterdir() if path.suffix == ".html"] == []
    assert provider.responses == [] and provider.turns == []


# --------------------------------------------------------------------------
# Criterion 2 — an ambiguous request costs one round of questions, then runs.
# --------------------------------------------------------------------------


def test_cli_run_answers_one_clarifying_question_and_then_completes(
    monkeypatch: pytest.MonkeyPatch, offline_run: OfflineRun
) -> None:
    """#10 at the CLI boundary: the command asks once through whoever it offered, feeds the
    answer back, and the run proceeds to a report. The `run_once` arm is `test_app.py`'s.

    `_asker` is patched rather than the streams: `CliRunner` supplies a pipe for stdin, so
    the command would otherwise correctly decide there is nobody to ask.
    """
    question = Question(
        kind=QuestionKind.SINGLE_CHOICE,
        text="Which metric should the chart show?",
        choices=["revenue", "profit"],
    )
    provider = _provider(LINEAR)
    provider.responses.insert(0, PlannerDraft(action=PlannerAction.CLARIFY, questions=[question]))
    offline_run(provider)
    asker = ScriptedAsker(answers=["revenue"])
    monkeypatch.setattr(cli_app, "_asker", lambda: asker)

    result = runner.invoke(app, ["run", "Make a chart of performance"])

    assert result.exit_code == ExitCode.SUCCESS
    assert asker.asked == [question]
    assert SUMMARY in result.stdout


# --------------------------------------------------------------------------
# Criterion 3 — the failure paths, all bounded: never a hang, never a traceback.
# --------------------------------------------------------------------------


def _ghost_plan() -> PlannerDraft:
    """A well-formed draft nothing can run: it consumes a step that does not exist.

    Well-formed rather than `None`, so the run really reaches `validate` — a refusal
    short-circuits before the plan is ever checked.
    """
    return PlannerDraft(
        action=PlannerAction.PLAN,
        subtasks=[
            SubtaskDraft(
                id="fetch_quarterly_financials",
                role=AgentRole.DATA_RETRIEVAL,
                instruction="Load revenue for the last three quarters.",
                inputs=["ghost"],
                depends_on=["ghost"],
            )
        ],
    )


def test_cli_run_with_an_unusable_plan_exits_task_failure_without_a_traceback(
    offline_run: OfflineRun,
) -> None:
    """Every attempt is well-formed and unusable, so planning gives up: exit 5, the reason
    on stderr, and nothing half-written for a pipe to parse (§5, §8)."""
    provider = FakeProvider(responses=[_ghost_plan() for _ in range(STRUCTURED_ATTEMPTS)])
    offline_run(provider)

    result = runner.invoke(app, ["run", LINEAR.prompt])

    assert result.exit_code == ExitCode.TASK_FAILURE
    assert result.stdout == ""
    assert "no usable plan" in result.stderr
    assert "Traceback" not in result.stderr
    assert provider.responses == []  # it really did spend every attempt


@pytest.mark.asyncio
async def test_a_malformed_chart_draft_fails_one_step_and_the_run_still_reports(
    monkeypatch: pytest.MonkeyPatch, offline_run: OfflineRun
) -> None:
    """Unusable output mid-run costs its own subtask and nothing else: the earlier steps
    keep their artifacts and the report is written over them (#8)."""
    # One dispatch per subtask, so what is under test is the structured call's retries and
    # not the engine's.
    monkeypatch.setenv("SUBTASK_ATTEMPTS", "1")
    provider = _provider(LINEAR)
    provider.responses[1:2] = [None] * STRUCTURED_ATTEMPTS  # the chart draft, refused every time
    offline_run(provider)

    async with asyncio.timeout(BOUND_SECONDS):
        state = await run_once(LINEAR.prompt)

    assert [subtask.id for subtask in state.failed_subtasks] == [LINEAR_CHART.id]
    assert state.failed
    assert set(state.artifacts) == {LINEAR_FETCH.id, LINEAR_ANALYSIS.id}
    report = state.final_result
    assert report is not None
    assert report.executive_summary == SUMMARY
    assert report.chart is None


@pytest.mark.asyncio
async def test_a_runaway_script_is_killed_and_costs_only_its_own_step(
    monkeypatch: pytest.MonkeyPatch, offline_run: OfflineRun
) -> None:
    """The analytics script never ends, so the executor kills it on its clock: the subtask
    fails, the run returns rather than hanging, and the model is told why (§6)."""
    monkeypatch.setenv("SUBTASK_ATTEMPTS", "1")
    # The class reference in `toolsets` is the seam: `Config` has no field for the
    # executor's clock, and `DEFAULT_TIMEOUT` binds at def time, so patching the module
    # constant would do nothing. The real class, only built with a shorter clock.
    monkeypatch.setattr(
        toolsets, "RunPythonTool", lambda store: RunPythonTool(store, timeout=EXECUTOR_TIMEOUT)
    )
    provider = FakeProvider(
        responses=[LINEAR.draft(), _report_draft(_pointer(LINEAR_FETCH))],
        turns=[
            *_csv_turns(),
            *_python_turns("while True:\n    pass\n", [_pointer(LINEAR_FETCH)]),
        ],
    )
    offline_run(provider)

    async with asyncio.timeout(BOUND_SECONDS):
        state = await run_once(LINEAR.prompt)

    assert [subtask.id for subtask in state.failed_subtasks] == [LINEAR_ANALYSIS.id]
    assert state.failed
    assert state.artifacts == {LINEAR_FETCH.id: _pointer(LINEAR_FETCH)}
    assert state.final_result is not None  # the retrieval step is still reported
    # The surfaced failure is the worker's ("computed nothing"), so the timeout itself is
    # read off the transcript — the one place it reached the model.
    returned = provider.send_calls[-1].messages[-1].tool_results[0].content
    assert f"killed after {EXECUTOR_TIMEOUT:g} seconds" in returned


def test_cli_run_whose_synthesis_is_refused_degrades_to_the_ledger_and_exits_zero(
    offline_run: OfflineRun,
) -> None:
    """Nothing about the *work* failed, so a report the model would not write is a degraded
    summary at exit 0 — the quiet path, worth pinning because nothing else says it."""
    provider = _provider(LINEAR)
    provider.responses[-1:] = [None] * STRUCTURED_ATTEMPTS  # the report call, refused every time
    offline_run(provider)

    result = runner.invoke(app, ["run", LINEAR.prompt])

    assert result.exit_code == ExitCode.SUCCESS
    assert "No synthesis was available for this run" in result.stdout
    assert LINEAR_CHART.id in result.stdout  # what did finish is still named


# --------------------------------------------------------------------------
# Criterion 4 — Ctrl-C mid-run. The line-discipline half is `test_chat.py`'s,
# which owns `ConsoleChat`; `CliRunner` cannot reach it (stdin is a pipe, so no
# chat is built at all).
# --------------------------------------------------------------------------


def test_cli_run_interrupted_inside_a_step_exits_130_and_releases_the_terminal(
    monkeypatch: pytest.MonkeyPatch, offline_run: OfflineRun
) -> None:
    """Ctrl-C lands inside a worker, so it unwinds a `TaskGroup` with work in flight: the
    dashboard's `Live` is stopped, the provider released, and the boundary maps it to 130
    with no traceback (§8, §10)."""
    provider = FakeProvider(responses=[LINEAR.draft()], turns=[KeyboardInterrupt()])
    offline_run(provider)
    # Forced, or `_render_mode` picks `PLAIN` for the runner's pipe and no `Live` is opened
    # — leaving nothing to leave behind (`test_cli.py` gives the same reason).
    monkeypatch.setattr(err_console, "_force_terminal", True, raising=False)

    result = runner.invoke(app, ["run", LINEAR.prompt])

    assert result.exit_code == ExitCode.INTERRUPTED
    assert result.stdout == ""
    # `in`, never `==`: asyncio may log an unretrieved task exception onto the real stderr.
    assert "Interrupted." in result.stderr
    assert "Traceback" not in result.stderr
    # Rich pops the stack in `Live.stop()`. Read through the private for the reason
    # `test_app.py`'s bounds test gives: a region still owning the terminal is invisible
    # until the user's shell has no cursor.
    assert err_console._live_stack == []
    assert "\x1b[?25h" in result.stderr  # and the cursor was shown again
    assert provider.closed


# --------------------------------------------------------------------------
# Criterion 6 (stretch, #12 landed) — the mid-run pause reshapes what is left,
# with the real workers.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_interrupt_replans_what_is_left_and_only_the_new_step_runs(
    offline_run: OfflineRun,
) -> None:
    """The key lands once the first step has finished; the orchestrator replaces the two
    unfinished steps with one, and the replaced chart step never runs."""
    replan = InterruptDraft(
        action=InterruptAction.REPLAN,
        reply="Dropping the chart and summarising the trend instead.",
        subtasks=[
            SubtaskDraft(
                id=REPLANNED_STEP,
                role=AgentRole.ANALYTICS,
                instruction="Summarise the quarterly revenue trend in one paragraph.",
                inputs=[LINEAR_FETCH.id],
                depends_on=[LINEAR_FETCH.id],
            )
        ],
    )
    provider = FakeProvider(
        # The pause's reply sits between the plan and the report, where the run asks for it.
        responses=[
            LINEAR.draft(),
            replan,
            _report_draft(f"{ARTIFACT_PREFIX}{REPLANNED_STEP}.json"),
        ],
        turns=[
            *_csv_turns(),
            *_python_turns(
                _COUNT_ROWS.format(dataset=_file(LINEAR_FETCH)), [_pointer(LINEAR_FETCH)]
            ),
        ],
    )
    offline_run(provider)
    # Keyed on the provider's call count, which is what makes the pause land after step one
    # rather than wherever the scheduler happened to be.
    chat = ScriptedChat(
        messages=["drop the chart, just summarise it"],
        armed=lambda: len(provider.send_calls) >= len(_csv_turns()),
    )

    async with asyncio.timeout(BOUND_SECONDS):
        state = await run_once(LINEAR.prompt, chat=chat)

    assert chat.sessions == 1
    assert chat.said == [replan.reply]
    assert state.plan is not None
    assert [subtask.id for subtask in state.plan.subtasks] == [LINEAR_FETCH.id, REPLANNED_STEP]
    assert set(state.artifacts) == {LINEAR_FETCH.id, REPLANNED_STEP}
    assert not state.failed
    report = state.final_result
    assert report is not None
    assert report.chart is None  # the replaced step is the one that would have drawn it


# --------------------------------------------------------------------------
# The un-stubbed CLI path, which every criterion above rides on. Every other
# `run` test replaces `run_once`; only these two pin the real Typer -> app ->
# agents wiring.
# --------------------------------------------------------------------------


def test_cli_run_prints_the_whole_report_on_stdout_and_progress_on_stderr(
    offline_run: OfflineRun,
) -> None:
    """The happy path with nothing stubbed but the provider port: exit 0, the report and
    the trace on stdout, and the run's progress kept off it (§5)."""
    provider = _provider(LINEAR)
    offline_run(provider)

    result = runner.invoke(app, ["run", LINEAR.prompt])

    assert result.exit_code == ExitCode.SUCCESS
    assert SUMMARY in result.stdout
    assert CHART_CATEGORY in result.stdout  # the drawing, not just a pointer to one
    for step in (LINEAR_FETCH, LINEAR_ANALYSIS, LINEAR_CHART):
        assert f"done     {step.id}" in result.stdout
    assert SUMMARY not in result.stderr


def test_cli_run_output_json_emits_exactly_one_document(offline_run: OfflineRun) -> None:
    """The same run under `-o json`: `json.loads` is the assertion, because it rejects a
    stray progress line or a second document as trailing data (§5)."""
    provider = _provider(LINEAR)
    offline_run(provider)

    result = runner.invoke(app, ["run", LINEAR.prompt, "-o", "json"])

    assert result.exit_code == ExitCode.SUCCESS
    document = json.loads(result.stdout)
    assert document["request"] == LINEAR.prompt
    assert document["status"] == "completed"
    assert document["report"]["executive_summary"] == SUMMARY
    assert [subtask["id"] for subtask in document["subtasks"]] == [
        subtask.id for subtask in LINEAR.draft().subtasks
    ]
    assert all(subtask["status"] == SubtaskStatus.DONE for subtask in document["subtasks"])
