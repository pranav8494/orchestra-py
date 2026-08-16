"""The retry instruction shared by structured calls. Text only — no logic (§11).

Schema-agnostic on purpose: an agent with something specific to say ships its own, as
`prompts/planner.py` does.
"""

REFORMAT_INSTRUCTION = """\
Your previous reply was rejected before it could be used; the problem is described \
below. Return a corrected reply in full, matching the schema exactly.\
"""
