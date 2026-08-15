"""Role workers: one per `AgentRole`, all behind the `Worker` port in `base.py`.

Phase A ships only `EchoWorker`, which proves the engine's dispatch and event contract
without a model. The real Data Retrieval, Analytics and Visualization agents (#5-#7)
replace it one at a time by implementing the same port (§6).
"""
