"""Turning a finished run into text for stdout, separate from rendering (§3.3).

Formatting and rendering are split deliberately: this module returns a string and knows
nothing about Rich or terminals, so it is testable without a console. The `text | json`
switch and the structured final report land here with #8; Phase A needs only enough to
show that the loop ran and what each step produced.
"""

from orchestra.core.state import TaskState


def format_run_summary(state: TaskState) -> str:
    """One line per subtask: status, id, and the artifact it produced.

    Args:
        state: the finished run's ledger.

    Returns:
        The summary, without a trailing newline.
    """
    if state.plan is None:
        return "No plan was produced."
    return "\n".join(
        f"{subtask.status.value:<8} {subtask.id}  {subtask.output_pointer or '-'}"
        for subtask in state.plan.subtasks
    )
