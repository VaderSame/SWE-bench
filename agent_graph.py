# agent_graph.py
import json
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from container import git_diff
from tools import ALL_TOOLS, EDIT_OK_PREFIX

load_dotenv()

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Turn at which we force edit-only mode if no successful edit has happened.
FORCE_EDIT_AFTER = 20
# How many times the model may view the *exact same* read-only call before we
# warn (+1) and then restrict to edit-only mode (+2 above warn).
VIEW_REPEAT_WARN     = 2   # warn on the Nth repeat
VIEW_REPEAT_RESTRICT = 5   # restrict on this many repeats

# Failing edit calls: after this many failures with identical args, inject guidance.
EDIT_FAIL_WARN = 2

# Agent turns allowed after the first successful edit before the run ends.
POST_EDIT_TURNS = 8

# Context-window management.
KEEP_FULL_VIEWS = 15
KEEP_FULL_OTHER = 5
STALE_OUTPUT_CHARS = 1500

# Tools that are ALWAYS executed and never blocked/deduplicated.
EDIT_TOOLS = {"replace_lines"}
# Read-only tools subject to repeat-tracking.
READ_TOOLS = {"view_file", "search_code", "repo_map", "explore_directory",
              "run_python_repro", "run_bash"}

DUP_PREFIX = "DUPLICATE CALL REJECTED"

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


class SWEBenchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    max_iterations: int
    patch: str


SYSTEM_PROMPT = """You are an autonomous software engineer fixing a bug in an open-source Python repository.

Rules:
1. Use view_file to read code. Always view at least 60 lines at a time.
2. Use run_python_repro to reproduce the bug. The script MUST assert and print PASS or FAIL.
3. Once you locate the faulty line, call replace_lines IMMEDIATELY.
   replace_lines(file_path, start_line, end_line, new_str) replaces lines
   start_line..end_line (1-indexed, inclusive). new_str must have correct indentation
   and must NOT include the lines just outside that range.
4. After a successful edit, run your repro to verify. PASS → done. FAIL → correct the edit.
5. Do NOT keep re-reading lines you have already seen. Locate the bug, fix it, verify.
"""

_base_llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base=os.environ.get("OPENROUTER_SERVER"),
    model_name=os.environ.get("MODEL_ID"),
    temperature=0.2,
    max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "3000")),
    timeout=float(os.getenv("LLM_TIMEOUT", "180")),
    max_retries=1,
)
llm           = _base_llm.bind_tools(ALL_TOOLS)
llm_edit_only = _base_llm.bind_tools([t for t in ALL_TOOLS if t.name in EDIT_TOOLS])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _call_sig(call: dict) -> str:
    return call["name"] + ":" + json.dumps(call["args"], sort_keys=True, default=str)


def _is_successful_edit(m: BaseMessage) -> bool:
    return (
        isinstance(m, ToolMessage)
        and m.name in EDIT_TOOLS
        and str(m.content).startswith(EDIT_OK_PREFIX)
    )


def _successful_edits(messages: list[BaseMessage]) -> int:
    return sum(1 for m in messages if _is_successful_edit(m))


def _turns_since_first_edit(messages: list[BaseMessage]) -> int:
    started, n = False, 0
    for m in messages:
        if _is_successful_edit(m):
            started = True
        elif started and isinstance(m, AIMessage):
            n += 1
    return n


def _read_call_counts(messages: list[BaseMessage]) -> dict[str, int]:
    """Count read-only tool calls since the last successful edit (or from start)."""
    counts: dict[str, int] = {}
    for m in messages:
        if _is_successful_edit(m):
            counts = {}
        elif isinstance(m, AIMessage):
            for c in m.tool_calls:
                if c["name"] in READ_TOOLS:
                    sig = _call_sig(c)
                    counts[sig] = counts.get(sig, 0) + 1
    return counts


def _failed_edit_counts(messages: list[BaseMessage]) -> dict[str, int]:
    """
    Count how many times each replace_lines signature has FAILED (not succeeded).
    Resets after a successful edit. Uses tool_call_id to correlate calls → results.
    """
    counts: dict[str, int] = {}
    pending: dict[str, str] = {}   # tool_call_id -> call signature

    for m in messages:
        if isinstance(m, AIMessage):
            for c in m.tool_calls:
                if c["name"] in EDIT_TOOLS:
                    pending[c["id"]] = _call_sig(c)
        elif isinstance(m, ToolMessage) and m.name in EDIT_TOOLS:
            sig = pending.pop(m.tool_call_id, None)
            if sig is not None:
                if str(m.content).startswith(EDIT_OK_PREFIX):
                    counts = {}          # any success resets all failure counts
                else:
                    counts[sig] = counts.get(sig, 0) + 1
    return counts


def _last_view_content(messages: list[BaseMessage]) -> str:
    """Content of the most recent successful view_file ToolMessage."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == "view_file":
            c = str(m.content)
            if not c.startswith(DUP_PREFIX):
                return c
    return ""


def _compact_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Shallow-copy of history with stale tool outputs truncated."""

    def is_real(m):
        return isinstance(m, ToolMessage) and not str(m.content).startswith(DUP_PREFIX)

    views  = [i for i, m in enumerate(messages) if is_real(m) and m.name == "view_file"]
    others = [i for i, m in enumerate(messages) if is_real(m) and m.name not in ("view_file",)]
    keep_full = set(views[-KEEP_FULL_VIEWS:]) | set(others[-KEEP_FULL_OTHER:])

    out = []
    for i, m in enumerate(messages):
        if (
            isinstance(m, ToolMessage)
            and i not in keep_full
            and isinstance(m.content, str)
            and len(m.content) > STALE_OUTPUT_CHARS
        ):
            m = m.model_copy(
                update={"content": m.content[:STALE_OUTPUT_CHARS] + "\n... [older output truncated]"}
            )
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def agent_node(state: SWEBenchState) -> dict:
    messages = _compact_history(state["messages"])
    no_edit  = _successful_edits(state["messages"]) == 0
    counts   = _read_call_counts(state["messages"])
    max_reps = max(counts.values(), default=0)

    force_edit = no_edit and (
        state["iteration"] >= FORCE_EDIT_AFTER
        or max_reps >= VIEW_REPEAT_RESTRICT
    )

    if force_edit:
        last_view = _last_view_content(state["messages"])
        ctx = f"\nFor reference, the last code you viewed:\n{last_view}\n" if last_view else ""
        msg = (
            "You have spent enough turns reading code."
            f"{ctx}"
            "ONLY `replace_lines` is available now. Apply your fix immediately using "
            "the line numbers shown above."
        )
        messages = messages + [HumanMessage(content=msg)]
        model = llm_edit_only
    else:
        model = llm
        if not no_edit:
            messages = messages + [
                HumanMessage(
                    content="An edit has been applied. Verify with run_python_repro (assert + "
                    "print PASS or FAIL). PASS → done. FAIL → fix the edit."
                )
            ]

    response = model.invoke(messages)
    return {"messages": [response], "iteration": state["iteration"] + 1}


def tools_node(state: SWEBenchState) -> dict:
    messages = state["messages"]
    last     = messages[-1]

    read_counts = _read_call_counts(messages[:-1])
    fail_counts = _failed_edit_counts(messages[:-1])

    results = []
    for call in last.tool_calls:
        tool = TOOL_MAP.get(call["name"])
        try:
            content = str(tool.invoke(call["args"])) if tool else f"Unknown tool {call['name']}"
        except Exception as e_:
            content = f"Tool error: {e_}"

        if call["name"] in READ_TOOLS:
            sig = _call_sig(call)
            n   = read_counts.get(sig, 0)
            if n >= VIEW_REPEAT_WARN:
                nth = {2: "3rd", 3: "4th", 4: "5th"}.get(n, f"{n+1}th")
                content += (
                    f"\n\n[WARNING: This is the {nth} time you called `{call['name']}` "
                    "with these exact arguments and no edit has been applied yet. "
                    "Stop re-reading and call `replace_lines` NOW.]"
                )
            read_counts[sig] = n + 1   # update so multiple calls in one turn are tracked

        elif call["name"] in EDIT_TOOLS:
            sig    = _call_sig(call)
            n_fail = fail_counts.get(sig, 0)
            if n_fail >= EDIT_FAIL_WARN and not content.startswith(EDIT_OK_PREFIX):
                content += (
                    f"\n\n[CRITICAL: This exact replace_lines call has failed {n_fail + 1} times. "
                    "• 'identical content' → those lines are already correct; the bug is ELSEWHERE. "
                    "Use view_file or search_code to look at a different function. "
                    "• SyntaxError → your new_str includes context lines that surround start_line..end_line. "
                    "Only include the lines that replace the range, nothing outside it.]"
                )

        results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))

    return {"messages": results}


def nudge_node(state: SWEBenchState) -> dict:
    return {
        "messages": [
            HumanMessage(
                content="You have not modified any files yet. "
                "Call `replace_lines` now to apply your fix."
            )
        ]
    }


def route_decision(state: SWEBenchState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return "extract_patch"

    if _turns_since_first_edit(state["messages"]) >= POST_EDIT_TURNS:
        return "extract_patch"

    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"

    if not git_diff().strip() and state["iteration"] < state["max_iterations"] - 2:
        return "nudge"

    return "extract_patch"


def extract_patch_node(state: SWEBenchState) -> dict:
    return {"patch": git_diff()}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
builder = StateGraph(SWEBenchState)
builder.add_node("agent",         agent_node)
builder.add_node("tools",         tools_node)
builder.add_node("nudge",         nudge_node)
builder.add_node("extract_patch", extract_patch_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_decision,
    {"tools": "tools", "nudge": "nudge", "extract_patch": "extract_patch"},
)
builder.add_edge("tools",         "agent")
builder.add_edge("nudge",         "agent")
builder.add_edge("extract_patch", END)

swe_agent = builder.compile()