"""Prompt registry: one module per agent, re-exported here (CONVENTIONS.md §3.3, §11).

The single import surface is what keeps prompt text out of the agents. Worker prompts
join this list with the agents that use them (#5-#7); when a shared preamble appears,
it becomes a `base` module the role modules compose with, not a copied block (§2).
"""

from orchestra.prompts.aggregator import SYSTEM_PROMPT as AGGREGATOR_SYSTEM_PROMPT
from orchestra.prompts.planner import REFORMAT_INSTRUCTION as PLANNER_REFORMAT_INSTRUCTION
from orchestra.prompts.planner import SYSTEM_PROMPT as PLANNER_SYSTEM_PROMPT

__all__ = ["AGGREGATOR_SYSTEM_PROMPT", "PLANNER_REFORMAT_INSTRUCTION", "PLANNER_SYSTEM_PROMPT"]
