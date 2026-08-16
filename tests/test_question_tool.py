"""Tests for the `ask_user` tool.

Nothing here reads stdin: `Asker` is a port, so `conftest.ScriptedAsker` answers from a
queue and records the typed questions it was handed (§12).
"""

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING

import pytest

from conftest import ScriptedAsker, tool_call, wait_until
from orchestra.tools.base import BaseTool, ToolCall
from orchestra.tools.question import DECLINED, TOOL_NAME, AskUserTool

# Ceiling on every wait here. Long enough not to flake on a loaded machine, short enough
# that a swallowed cancellation fails rather than hangs the suite.
TIMEOUT = 5.0


if TYPE_CHECKING:
    # Conformance is mypy's job: `BaseTool` is a plain Protocol, and `isinstance` would
    # compare attribute names only (§7).
    _ASK_USER_IS_A_TOOL: BaseTool = AskUserTool(ScriptedAsker())


def properties(schema: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """A JSON Schema's `properties` block, narrowed so a test can index into it."""
    block = schema["properties"]
    assert isinstance(block, dict)
    return block


def yes_no_call() -> ToolCall:
    """The simplest valid call, for the tests that are not about the arguments."""
    return tool_call(TOOL_NAME, kind="yes_no", text="Include 2024?")


def test_ask_user_info_publishes_the_question_model_as_its_schema() -> None:
    """The schema is `Question` itself, so no second params model can drift from it
    (§1.5). Asserted field by field rather than against `Question.model_json_schema()`,
    which is the expression `info()` returns."""
    spec = AskUserTool(ScriptedAsker()).info()

    assert spec.name == "ask_user"
    assert set(properties(spec.input_schema)) == {"kind", "text", "description", "choices"}
    # `extra="forbid"` reaches the model, so an invented `options` is rejected client-side
    # as well as by `run`.
    assert spec.input_schema["additionalProperties"] is False


def test_ask_user_info_description_says_when_not_to_ask() -> None:
    """The cost of this tool is a human turn-around, so the prompt has to spend most of
    its words fencing it off."""
    description = AskUserTool(ScriptedAsker()).info().description

    assert "Do not use it" in description
    assert "free_text" in description
    # The model must know a blank answer is an outcome, not a reason to ask again.
    assert "declined" in description


@pytest.mark.asyncio
async def test_ask_user_answer_is_returned_as_the_tool_output() -> None:
    asker = ScriptedAsker(answers=["2024 and 2025"])

    response = await AskUserTool(asker).run(
        tool_call(TOOL_NAME, kind="free_text", text="Which years?")
    )

    assert not response.is_error and not response.is_empty
    assert response.content == "2024 and 2025"


@pytest.mark.asyncio
async def test_ask_user_hands_the_asker_a_typed_question() -> None:
    """The `Asker` is handed a validated `Question`, never the raw arguments — the renderer
    branches on `kind` and must not have to parse a sentence (§7)."""
    asker = ScriptedAsker(answers=["Q1"])

    await AskUserTool(asker).run(
        tool_call(
            TOOL_NAME,
            kind="single_choice",
            text="Which quarter?",
            description="The report covers one.",
            choices=["Q1", "Q2"],
        )
    )

    asked = asker.asked[0]
    assert asked.kind == "single_choice"
    assert asked.text == "Which quarter?"
    assert asked.choices == ["Q1", "Q2"]


@pytest.mark.asyncio
async def test_ask_user_choice_question_without_choices_is_an_error_naming_choices() -> None:
    """The kind/choices rule is a model validator, so its message has to survive
    `format_validation_error` and reach the model as its retry (§6)."""
    asker = ScriptedAsker()

    response = await AskUserTool(asker).run(
        tool_call(TOOL_NAME, kind="single_choice", text="Which quarter?", choices=["Q1"])
    )

    assert response.is_error
    assert TOOL_NAME in response.content
    assert "choices" in response.content
    assert asker.asked == []  # nobody was prompted with a question that cannot be rendered


@pytest.mark.asyncio
async def test_ask_user_unknown_kind_is_an_error_naming_the_field() -> None:
    response = await AskUserTool(ScriptedAsker()).run(
        tool_call(TOOL_NAME, kind="dropdown", text="Which quarter?")
    )

    assert response.is_error
    assert "kind" in response.content


@pytest.mark.asyncio
async def test_ask_user_invented_argument_is_rejected() -> None:
    """`extra="forbid"`: an invented field means the schema drifted, not that it is free."""
    response = await AskUserTool(ScriptedAsker()).run(
        tool_call(TOOL_NAME, kind="yes_no", text="Include 2024?", required=True)
    )

    assert response.is_error
    assert "required" in response.content


@pytest.mark.asyncio
async def test_ask_user_blank_answer_is_empty_not_an_error() -> None:
    """Declining is an outcome. `is_error` would read to the model as "the tool broke" and
    buy a pointless retry of a question the user has already refused (§6)."""
    response = await AskUserTool(ScriptedAsker(answers=[""])).run(yes_no_call())

    assert response.is_empty
    assert not response.is_error
    # A sentence, not "": every other tool pairs `is_empty` with something to act on, and
    # the API rejects an empty `tool_result` block outright.
    assert response.content == DECLINED
    assert "declined" in response.content


@pytest.mark.asyncio
async def test_ask_user_whitespace_answer_is_empty() -> None:
    """Space-then-Enter is a decline as much as Enter is."""
    response = await AskUserTool(ScriptedAsker(answers=["   \n"])).run(yes_no_call())

    assert response.is_empty and not response.is_error


@pytest.mark.asyncio
async def test_ask_user_propagates_cancellation() -> None:
    """§10: `CancelledError` is the only thing that may leave `run`. A `try` widened to
    `BaseException` around the ask — or a handler turning cancellation into a decline —
    fails here."""
    blocker = asyncio.Event()
    asker = ScriptedAsker(answers=["never read"], blocker=blocker)
    task = asyncio.create_task(AskUserTool(asker).run(yes_no_call()))
    await wait_until(lambda: bool(asker.asked), what="the question to be put to the user")

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=TIMEOUT)
