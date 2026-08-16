"""Prompt registry: one module per agent, re-exported here (CONVENTIONS.md §3.3, §11)."""

from orchestra.prompts.aggregator import SYSTEM_PROMPT as AGGREGATOR_SYSTEM_PROMPT
from orchestra.prompts.analytics import SYSTEM_PROMPT as ANALYTICS_SYSTEM_PROMPT
from orchestra.prompts.data_retrieval import SYSTEM_PROMPT as DATA_RETRIEVAL_SYSTEM_PROMPT
from orchestra.prompts.planner import REFORMAT_INSTRUCTION as PLANNER_REFORMAT_INSTRUCTION
from orchestra.prompts.planner import SYSTEM_PROMPT as PLANNER_SYSTEM_PROMPT
from orchestra.prompts.structured import REFORMAT_INSTRUCTION as STRUCTURED_REFORMAT_INSTRUCTION
from orchestra.prompts.visualization import SYSTEM_PROMPT as VISUALIZATION_SYSTEM_PROMPT

__all__ = [
    "AGGREGATOR_SYSTEM_PROMPT",
    "ANALYTICS_SYSTEM_PROMPT",
    "DATA_RETRIEVAL_SYSTEM_PROMPT",
    "PLANNER_REFORMAT_INSTRUCTION",
    "PLANNER_SYSTEM_PROMPT",
    "STRUCTURED_REFORMAT_INSTRUCTION",
    "VISUALIZATION_SYSTEM_PROMPT",
]
