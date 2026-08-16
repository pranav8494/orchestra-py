"""The user turn every worker opens its subtask with.

Formatting lives here, not in `prompts/` (§11), and the untrusted text stays out of the
system prompt. How the inputs are shown is the caller's: a worker holding a tool that
opens artifacts names them, one without shows their contents.
"""

from collections.abc import Sequence

from orchestra.core.state import SubtaskContext


def build_briefing(context: SubtaskContext, input_lines: Sequence[str]) -> str:
    """Render the step, the request behind it, the inputs, then the clarifications.

    Args:
        input_lines: the inputs as the caller chose to show them. Empty drops the header.
    """
    lines = [
        f"Subtask: {context.subtask.instruction}",
        f"The request this serves: {context.user_request}",
    ]
    if input_lines:
        lines.append("Earlier steps produced:")
        lines += input_lines
    lines += [
        f"Clarification asked: {item.question}\nThe user answered: {item.answer}"
        for item in context.clarifications
    ]
    return "\n".join(lines)
