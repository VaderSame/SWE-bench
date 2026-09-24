# agent_graph.py
import json
import os
import subprocess
from pathlib import Path
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

from tools import ALL_TOOLS, EDIT_OK_PREFIX

load_dotenv()

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
NUDGE_AFTER = 8             # turn at which tool results carry an "edit now" reminder
FORCE_EDIT_AFTER = 12       # turn at which the toolset shrinks to edit_file only (no successful edit yet)
STUCK_DUPLICATES = 2        # consecutive rejected duplicates that trigger the same restriction
MAX_STUCK = 4               # consecutive rejected duplicates that end the run early
POST_EDIT_TURNS = 8         # agent turns allowed after the first successful edit, then the run ends
KEEP_FULL_VIEWS = 6         # most recent real view_file outputs sent in full
KEEP_FULL_OTHER = 2         # most recent real outputs of other tools sent in full
STALE_OUTPUT_CHARS = 300    # everything older is cut to this many characters
DUP_PREFIX = "DUPLICATE CALL REJECTED"

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


class SWEBenchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    workspace_dir: str
    iteration: int
    max_iterations: int
    patch: str


# Generic prompt: no hints about the specific bug, so results stay comparable.
SYSTEM_PROMPT = """You are an autonomous software engineer fixing a bug in an open-source Python repository.
Your task is to sequence and call the given tools appropriately in order to solve the bug.

Available_tools:
{tools_schema}

Rules:
1. Never repeat a tool call with the same arguments unless a file changed since. Duplicates are rejected.
2. Do not inspect tiny windows of a file. View at least 60 lines at a time.
3. Keep reasoning out of code you pass to run_python_repro and edit_file. Write minimal code only.
4. edit_file replaces one exact occurrence of old_str with new_str. Copy old_str verbatim from a view_file
   output (without the line-number prefix), include a few surrounding lines so it is unique, and make sure
   new_str actually differs from old_str.
5. Once you have identified the faulty line, call edit_file immediately.
6. run_python_repro code must assert the expected behavior described in the issue and print PASS or FAIL.
   A repro that only prints values cannot verify anything.
7. After a successful edit, run your asserting repro to verify. If it prints PASS, conclude your response.
   If it prints FAIL, correct the edit. Do not go back to exploring.
"""

_base_llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base=os.environ.get("OPENROUTER_SERVER"),
    model_name=os.environ.get("MODEL_ID"),
    temperature=0.2,
    max_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "3000")),  # a runaway generation can look like a hang
    timeout=float(os.getenv("LLM_TIMEOUT", "180")),          # raise instead of waiting forever
    max_retries=1,
    # For a local vLLM server you can also try:
    # extra_body={"repetition_penalty": 1.05},
)
llm = _base_llm.bind_tools(ALL_TOOLS)
llm_edit_only = _base_llm.bind_tools([t for t in ALL_TOOLS if t.name == "edit_file"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _call_signature(call: dict) -> str:
    return call["name"] + ":" + json.dumps(call["args"], sort_keys=True, default=str)


def _is_successful_edit(m: BaseMessage) -> bool:
    return (
        isinstance(m, ToolMessage)
        and m.name == "edit_file"
        and str(m.content).startswith(EDIT_OK_PREFIX)
    )


def _successful_edits(messages: list[BaseMessage]) -> int:
    return sum(1 for m in messages if _is_successful_edit(m))


def _view_range(call: dict):
    a = call["args"]
    try:
        return a.get("file_path"), int(a.get("start_line") or 1), int(a.get("end_line") or 150)
    except (TypeError, ValueError):
        return None, 0, -1


def _seen_signatures(messages: list[BaseMessage]):
    """Earlier calls tagged with the 'edit epoch' they ran in.

    The epoch increases after every successful edit, so re-viewing a file or
    re-running a repro after the file changed is NOT a duplicate.
    Also collects the line ranges already shown per (epoch, file), so overlapping
    re-views can be rejected even when their arguments differ slightly.
    Returns (seen_set, ranges, current_epoch).
    """
    seen, ranges, epoch = set(), {}, 0
    for m in messages:
        if isinstance(m, AIMessage):
            for c in m.tool_calls:
                seen.add((epoch, _call_signature(c)))
                if c["name"] == "view_file":
                    path, s, e = _view_range(c)
                    if path and s <= e:
                        ranges.setdefault((epoch, path), []).append((s, e))
        elif _is_successful_edit(m):
            epoch += 1
    return seen, ranges, epoch


def _turns_since_first_edit(messages: list[BaseMessage]) -> int:
    started, n = False, 0
    for m in messages:
        if _is_successful_edit(m):
            started = True
        elif started and isinstance(m, AIMessage):
            n += 1
    return n


def _consecutive_duplicates(messages: list[BaseMessage]) -> int:
    n = 0
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            if str(m.content).startswith(DUP_PREFIX):
                n += 1
            else:
                break
    return n


def _compact_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Copy of the history with old tool outputs shortened. Stored state is untouched."""

    def is_real(m):
        return isinstance(m, ToolMessage) and not str(m.content).startswith(DUP_PREFIX)

    views = [i for i, m in enumerate(messages) if is_real(m) and m.name == "view_file"]
    others = [i for i, m in enumerate(messages) if is_real(m) and m.name != "view_file"]
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
    no_edit = _successful_edits(state["messages"]) == 0
    stuck = _consecutive_duplicates(state["messages"]) >= STUCK_DUPLICATES
    restrict = no_edit and (stuck or state["iteration"] >= FORCE_EDIT_AFTER)

    if restrict:
        # Server-independent: only edit_file exists, so further exploring is impossible.
        messages = messages + [
            HumanMessage(
                content="Exploration is over and only `edit_file` is available. Using the code you "
                "have already viewed, apply your best fix now. Copy old_str exactly from the viewed code."
            )
        ]
        model = llm_edit_only
    else:
        model = llm
        if not no_edit:
            messages = messages + [
                HumanMessage(
                    content="An edit has been applied. Do not explore further. Verify with a repro that "
                    "asserts the expected output from the issue and prints PASS or FAIL. If PASS, finish. "
                    "If FAIL, correct your edit."
                )
            ]
    response = model.invoke(messages)
    return {"messages": [response], "iteration": state["iteration"] + 1}


def tools_node(state: SWEBenchState) -> dict:
    """Runs tool calls, rejecting exact duplicates issued since the last successful edit."""
    messages = state["messages"]
    last = messages[-1]

    seen, ranges, epoch = _seen_signatures(messages[:-1])

    results = []
    for call in last.tool_calls:
        key = (epoch, _call_signature(call))
        covered = None
        if call["name"] == "view_file":
            path, s, e = _view_range(call)
            if path and s <= e:
                covered = next(
                    ((a, b) for a, b in ranges.get((epoch, path), []) if a <= s and e <= b), None
                )
        if key in seen:
            content = (
                f"{DUP_PREFIX}: `{call['name']}` was already run with these exact arguments and no edit "
                "has succeeded since, so the result is unchanged and is earlier in the conversation. "
                "Do something different. If you already know the faulty code, call edit_file now."
            )
        elif covered:
            content = (
                f"{DUP_PREFIX}: lines {s}-{e} of {path} were already shown in an earlier view_file output "
                f"(lines {covered[0]}-{covered[1]}) and the file has not changed since. Read that output "
                "instead, then act on it. If you know the faulty code, call edit_file now."
            )
        else:
            tool = TOOL_MAP.get(call["name"])
            try:
                content = str(tool.invoke(call["args"])) if tool else f"Unknown tool {call['name']}"
            except Exception as e_:
                content = f"Tool error: {e_}"
            if call["name"] == "view_file":
                path, s, e = _view_range(call)
                if path and s <= e:
                    ranges.setdefault((epoch, path), []).append((s, e))
        seen.add(key)
        results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))

    if results and state["iteration"] >= NUDGE_AFTER and _successful_edits(messages) == 0:
        results[-1] = results[-1].model_copy(
            update={
                "content": results[-1].content
                + f"\n\n[Reminder: {state['iteration']} of {state['max_iterations']} turns used "
                "and no edit has been applied. Stop exploring and call edit_file.]"
            }
        )
    return {"messages": results}


def nudge_node(state: SWEBenchState) -> dict:
    return {
        "messages": [
            HumanMessage(
                content="You have not modified any files yet. Call `edit_file` now to apply your fix."
            )
        ]
    }


def route_decision(state: SWEBenchState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return "extract_patch"

    if _consecutive_duplicates(state["messages"]) >= MAX_STUCK:
        return "extract_patch"

    if _turns_since_first_edit(state["messages"]) >= POST_EDIT_TURNS:
        return "extract_patch"

    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"

    workspace = Path(state["workspace_dir"])
    diff = subprocess.run(["git", "diff"], cwd=workspace, capture_output=True, text=True)
    if not diff.stdout.strip() and state["iteration"] < state["max_iterations"] - 2:
        return "nudge"

    return "extract_patch"


def extract_patch_node(state: SWEBenchState) -> dict:
    workspace = Path(state["workspace_dir"])
    subprocess.run(["git", "clean", "-f", "-q"], cwd=workspace)
    res = subprocess.run(["git", "diff"], cwd=workspace, capture_output=True, text=True)
    return {"patch": res.stdout.strip()}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
builder = StateGraph(SWEBenchState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tools_node)
builder.add_node("nudge", nudge_node)
builder.add_node("extract_patch", extract_patch_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_decision,
    {"tools": "tools", "nudge": "nudge", "extract_patch": "extract_patch"},
)
builder.add_edge("tools", "agent")
builder.add_edge("nudge", "agent")
builder.add_edge("extract_patch", END)

swe_agent = builder.compile()