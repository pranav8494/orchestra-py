# Research & Architectural Trade-off Analysis: Multi-Agent Task Solver

## Executive Summary

This document presents my comprehensive research analysis and architectural evaluation for designing and building a **Multi-Agent Task Solver**. My goal is to build an intelligent, multi-agent system that accepts plain-language business requests (e.g., *"Summarize the last 3 quarters' financial trends and create a chart"*), dynamically breaks down the task into subtasks, delegates execution to specialized AI agents, and returns a structured final result with real-time visibility.

Special focus is given to evaluating architectural patterns under my strict **8 to 10-hour implementation deadline**, ensuring high marks on handling ambiguity, avoiding common LLM pitfalls (hallucinations, repetition, infinite loops), and providing a clear path to stretch goals such as multi-turn refinement and live human-in-the-loop interaction.

---

## 1. Core Requirements & Success Criteria Analysis

| Requirement | Description | System Requirement / Architecture Need |
| :--- | :--- | :--- |
| **1. Input** | User inputs high-level natural language prompt. | Flexible interface accepting freeform text. |
| **2. Planning** | System decides agent allocation & task decomposition. | Dynamic planner (DAG generation / LLM Supervisor). |
| **3. Execution** | Specialized agents complete tasks using tools & shared state. | Context-isolated sub-agents with dedicated toolkits (Python REPL, Search). |
| **4. Aggregation** | System compiles outputs into a final report. | Synthesis step merging textual, numerical, and visual artifacts. |
| **5. Visibility** | Real-time progress updates per agent. | Event-driven status streaming / live UI dashboard. |
| **High Marks** | Handles ambiguous/incomplete requests via clarifying questions. | Pre-execution validation check & human-in-the-loop interrupt mechanism. |
| **Stretch Goals** | Mid-execution chat & multi-turn refinement. | Stateful graph with interruptible execution & delta replanning. |

---

## 2. Multi-Agent Architectural Patterns Evaluated

I evaluated four major structural design patterns for multi-agent systems:

```
Pattern A: Centralized Orchestrator       Pattern B: Peer-to-Peer Handoffs
         ┌─────────────┐                           ┌─────────────┐
         │ Orchestrator│                           │ Agent A     │
         └──────┬──────┘                           └──────┬──────┘
    ┌───────────┼───────────┐                             │ (handoff)
    ▼           ▼           ▼                             ▼
┌───────┐   ┌───────┐   ┌───────┐                  ┌─────────────┐
│Agent A│   │Agent B│   │Agent C│                  │ Agent B     │
└───────┘   └───────┘   └───────┘                  └─────────────┘

Pattern C: Blackboard System              Pattern D: Hierarchical Sub-teams
      ┌─────────────────┐                          ┌─────────────┐
      │ Shared State /  │                          │ Executive   │
      │ Event Log       │                          └──────┬──────┘
      └────────┬────────┘                             ┌───┴───┐
   ┌───────────┼───────────┐                          ▼       ▼
   ▼           ▼           ▼                      ┌──────┐ ┌──────┐
┌───────┐   ┌───────┐   ┌───────┐                 │Team A│ │Team B│
│Agent A│   │Agent B│   │Agent C│                 └──────┘ └──────┘
```

### Pattern A: Centralized Orchestrator / Supervisor
A dedicated central agent acts as the primary manager. It receives the user prompt, breaks it down into a sequence or Directed Acyclic Graph (DAG) of subtasks, invokes individual worker agents with targeted context, inspects outputs, and aggregates results.

* **Pros:** Highly predictable control flow, easy to enforce state validation, single point to inject retry limits and hallucination guardrails, straightforward to pause for clarification.
* **Cons:** The central supervisor can become a bottleneck or failure point if its prompt is poorly structured.

### Pattern B: Peer-to-Peer / Dynamic Handoffs
Agents pass control directly to one another via function calls without returning control to a central supervisor (e.g., OpenAI Swarm pattern).

* **Pros:** Low overhead, natural flow for simple sequential workflows or conversational routing.
* **Cons:** High risk of infinite loops (Agent A -> Agent B -> Agent A), difficult to maintain global context, harder to show structured top-level progress to the user.

### Pattern C: Blackboard / Shared Event Log
Agents communicate asynchronously by reading and writing to a shared data store (or event queue) without direct execution calls.

* **Pros:** Extremely decoupled, highly scalable, excellent for mass parallel processing.
* **Cons:** High infrastructure complexity (redis/database setup), non-deterministic execution order, difficult to complete in under 10 hours.

### Pattern D: Hierarchical (Team of Teams)
Tree structure where top-level supervisors delegate to sub-supervisors, who manage specialized agent pools.

* **Pros:** Solves severe cognitive load issues in massive multi-domain enterprise applications.
* **Cons:** Over-engineered for standard business automation; introduces unnecessary prompt chaining latency.

---

## 3. Deep Dive: Shared Context Architecture

In my chosen **Centralized Orchestrator + State Graph** architecture, context sharing is managed through a **Centralized State Schema** rather than passing full conversational histories between agents.

```
                  ┌─────────────────────────────────┐
                  │      Shared State Schema        │
                  │  (TypedDict / Pydantic Model)   │
                  └────────────────┬────────────────┘
                                   │
             ┌─────────────────────┼─────────────────────┐
             │ Reads / Writes      │ Reads / Writes      │ Reads / Writes
             ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │ DataFetchAgent  │   │ AnalyticsAgent  │   │ VizAgent        │
    └─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Key Mechanisms:

1. **Centralized Memory Ledger (`TaskState`):** A typed global state object stores task specifications, agent outputs, execution logs, and pointers to external data.
2. **Selective Context Isolation:** To prevent token bloat and context degradation, worker agents receive *only* the specific state slice required for their subtask (e.g., `AnalyticsAgent` receives only `artifacts["raw_financial_data"]`).
3. **Pointer-Based Artifact Storage:** Large datasets, DataFrames, and chart files are written to a local `artifacts/` directory; agents exchange string keys (e.g., `"artifact:q1_q3_financials.csv"`) in state, never raw blobs.
4. **Parallel Dispatch:** The plan is a DAG. Subtasks with satisfied dependencies run concurrently under `asyncio.TaskGroup` with a bounded semaphore.

### Proving the Planner Is Dynamic

A three-role pipeline can look dynamic while always emitting the same DAG. Three bundled scenarios show the plan responding to the shape of the request:

| Scenario | Prompt | Required plan shape |
| :--- | :--- | :--- |
| **Linear** | *"Summarize the last 3 quarters' financial trends and create a chart"* | 3 steps, sequential |
| **Fan-out** | *"Compare the last 3 quarters against the same quarters last year and chart both"* | 2 independent retrievals running **concurrently**, then Analytics, then Visualization |
| **Role omission** | *"Summarize the revenue trend in one paragraph"* | 2 steps, **no** Visualization subtask |

Asserted as planner unit tests on plan shape, and shown in the demo — fan-out is where the dashboard displays two agents live at once.

---

## 4. Deep Dive: Live Conversation Interruption & Mid-Execution Replanning

To satisfy the stretch goal of allowing live mid-execution user chats, my system implements an **Async Event Loop with Interruption Handling**.

```
                  ┌─────────────────────────────────────────┐
                  │          Async Execution Loop           │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                       [Check Interrupt Queue / Stdin]
                                       │
                  ┌────────────────────┴────────────────────┐
                  │                                         │
       [No User Interruption]                    [User Interruption Sent]
                  │                                         │
                  ▼                                         ▼
      Execute Next Planned Step                Pause Graph State Execution
      (e.g., Run Analytics Agent)              & Route Payload to Orchestrator
                                                            │
                                                            ▼
                                               Orchestrator System Prompt
                                               (Evaluate & Re-plan Graph)
```

### Orchestrator Interrupt System Prompt Specification

```text
ROLE: Lead Task Orchestrator. The user interrupted the current execution with a new constraint.

CURRENT STATE & CONTEXT:
Original Goal: {state.user_request}
Current Step: Step {state.current_step} of {len(state.plan)}
Completed Artifacts: {state.format_completed_artifacts()}
Remaining Plan: {state.format_remaining_plan()}

USER INTERRUPT MESSAGE: "{user_interrupt_message}"

INSTRUCTIONS:
1. Re-evaluate remaining steps given completed artifacts and the user's new constraint.
2. Return JSON with action: "REPLAN" | "RESTART_STEP" | "CLARIFY" | "CONTINUE".
3. Provide updated remaining plan without re-running completed steps.
```

---

## 5. Guardrails: Hallucination, Repetition, Runaway Loops

The rubric names these explicitly, so each gets a mechanism rather than a promise.

| Failure | Mechanism |
| :--- | :--- |
| **Hallucinated figures** | Every number in the final report must trace to an artifact pointer. Workers return `{value, source_pointer}`; aggregation drops any claim with no backing pointer. |
| **Malformed LLM output** | All plans, subtask results, and clarifications validate through Pydantic before touching state. Invalid output retries once with the validation error fed back; then fails the subtask. |
| **Repetition** | Loop detector: SHA-256 signature over `(tool_name, input, output)` per step, sliding window of 10, trip if any signature repeats more than 5 times. |
| **Runaway execution** | Per-subtask attempt cap (3) and global step budget (15). Exceeding either ends the run with a partial-results report, never a hang. |
| **Question loops** | At most one clarification round per run. |

---

## 6. Interface & UX Strategy: Web vs. Terminal CLI

A major design consideration for my **8 to 10-hour implementation budget** was deciding where to spend engineering effort: backend multi-agent orchestration vs. frontend presentation.

```
       [ 8–10 Hour Engineering Budget Allocation ]

   Option 1: Web Stack (FastAPI + React/WebSockets)
   ├── Multi-Agent Logic & Prompts   [ 40% ]
   ├── API Setup & Async Event Bus   [ 20% ]
   └── React UI, CSS, WebSocket Bugs [ 40% ]  <-- High Risk Area

   Option 2: Terminal TUI / CLI Stack (Rich / Typer)
   ├── Multi-Agent Logic & Prompts   [ 70% ]  <-- Maximize Quality & High Marks
   ├── Tool Integrations & Python REPL[ 15% ]
   └── Rich Terminal UI / Progress   [ 15% ]  <-- Low Effort, High Polish
```

---

## 7. Architectural Trade-off Matrix

| Feature / Criteria | Centralized Orchestrator + CLI | P2P Handoffs + CLI | Centralized + Web UI | Blackboard + Web UI |
| :--- | :--- | :--- | :--- | :--- |
| **Est. Build Time** | **6 – 7 Hours** | 5 – 6 Hours | 9 – 11 Hours | 12+ Hours |
| **Control & Determinism** | **High** | Low | High | Medium |
| **Loop & Hallucination Prevention**| **Very Easy** | Hard | Very Easy | Hard |
| **Clarification Interrupt Support** | **Built-in (`Rich.prompt`)**| Hard | Moderate (WebSockets) | Complex |
| **UI Polish vs. Effort Ratio** | **Exceptional** | Moderate | High (if finished) | High |
| **Risk of Scope Creep / Overrun** | **Very Low** | Low | High | Very High |

---

## 8. Detailed Implementation Roadmap (8-Hour Timeline)

The assignment allows 24 hours; I am budgeting 8–10 of focused build time and spending the rest on
documentation and the demo. Clarification lands before the stretch goals — it is the graded feature.

```
 Hour 0 - 2: Core Architecture & State Management
 ├── Define TaskState schema (Pydantic) + artifact store
 └── Implement Orchestrator prompt & initial task breakdown logic

 Hour 2 - 4: Specialized Worker Agents & Tools
 ├── Data Retrieval Agent (mock CSV + canned search)
 ├── Analytics Agent (subprocess Python executor)
 └── Visualization Agent (Plotly file + ASCII chart)

 Hour 4 - 5.5: Guardrails & Clarifying Questions
 ├── Schema validation, retry caps, step budget, loop detector
 └── Clarification round for ambiguous requests  <-- graded "High Marks" feature

 Hour 5.5 - 7: CLI Interface & Real-time Progress
 ├── Build Rich Live Dashboard (Tables, Spinners, Panel outputs)
 └── Hook execution events to CLI renderer

 Hour 7 - 8.5: Stretch Goals & Refinement
 ├── Mid-execution interruption handling & dynamic replanning
 └── End-to-end testing with sample business scenarios
```

---

## 9. Conclusion

By pairing a **Centralized Orchestrator Pattern** with selective context isolation and an interruptible state graph, my system addresses context limitations, handles mid-execution user chat gracefully, and prevents hallucinations. Using a **Rich Interactive CLI Interface** ensures full feature delivery well within my 8 to 10-hour implementation budget.
