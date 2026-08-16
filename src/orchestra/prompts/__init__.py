"""Prompt registry: one module per agent, re-exported here (CONVENTIONS.md §3.3, §11)."""

from orchestra.prompts.aggregator import SYSTEM_PROMPT as AGGREGATOR_SYSTEM_PROMPT
from orchestra.prompts.analytics import SYSTEM_PROMPT as ANALYTICS_SYSTEM_PROMPT
from orchestra.prompts.data_retrieval import SYSTEM_PROMPT as DATA_RETRIEVAL_SYSTEM_PROMPT
from orchestra.prompts.interrupt import REFORMAT_INSTRUCTION as INTERRUPT_REFORMAT_INSTRUCTION
from orchestra.prompts.interrupt import SITUATION as INTERRUPT_SITUATION
from orchestra.prompts.interrupt import SYSTEM_PROMPT as INTERRUPT_SYSTEM_PROMPT
from orchestra.prompts.planner import AVAILABLE_DATA as PLANNER_AVAILABLE_DATA
from orchestra.prompts.planner import CLARIFICATION_PREAMBLE as PLANNER_CLARIFICATION_PREAMBLE
from orchestra.prompts.planner import CLARIFY_SPENT as PLANNER_CLARIFY_SPENT
from orchestra.prompts.planner import CLARIFY_UNANSWERED as PLANNER_CLARIFY_UNANSWERED
from orchestra.prompts.planner import NO_DATA_LISTED as PLANNER_NO_DATA_LISTED
from orchestra.prompts.planner import REFORMAT_INSTRUCTION as PLANNER_REFORMAT_INSTRUCTION
from orchestra.prompts.planner import SYSTEM_PROMPT as PLANNER_SYSTEM_PROMPT
from orchestra.prompts.structured import REFORMAT_INSTRUCTION as STRUCTURED_REFORMAT_INSTRUCTION
from orchestra.prompts.visualization import SYSTEM_PROMPT as VISUALIZATION_SYSTEM_PROMPT

__all__ = [
    "AGGREGATOR_SYSTEM_PROMPT",
    "ANALYTICS_SYSTEM_PROMPT",
    "DATA_RETRIEVAL_SYSTEM_PROMPT",
    "INTERRUPT_REFORMAT_INSTRUCTION",
    "INTERRUPT_SITUATION",
    "INTERRUPT_SYSTEM_PROMPT",
    "PLANNER_AVAILABLE_DATA",
    "PLANNER_CLARIFICATION_PREAMBLE",
    "PLANNER_CLARIFY_SPENT",
    "PLANNER_CLARIFY_UNANSWERED",
    "PLANNER_NO_DATA_LISTED",
    "PLANNER_REFORMAT_INSTRUCTION",
    "PLANNER_SYSTEM_PROMPT",
    "STRUCTURED_REFORMAT_INSTRUCTION",
    "VISUALIZATION_SYSTEM_PROMPT",
]
