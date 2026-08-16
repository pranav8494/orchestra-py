"""Role workers: one per `AgentRole`, all behind the `Worker` port in `base.py`.

`EchoWorker` proved the engine's dispatch and event contract without a model, and still
stands in for the roles Phase B has not reached. `DataRetrievalWorker` is the first real
one (#5); Analytics and Visualization (#6, #7) replace their stubs the same way, by
implementing the same port (§6).
"""
