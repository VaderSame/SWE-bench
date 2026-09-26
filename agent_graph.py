# agent_graph.py
import json
import os
import posixpath
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

import container as ct
from container import git_diff
from tools import ALL_TOOLS, EDIT_OK_PREFIX

load_dotenv()

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Turn at which we force edit-only mode if no successful edit has happened.
FORCE_EDIT_AFTER = 18
# How many times the model may view the *exact same* read-only call before we
# warn (+1) and then restrict to edit-only mode (+2 above warn).
VIEW_REPEAT_WARN     = 1   # warn on the 2nd exact repeat
VIEW_REPEAT_RESTRICT = 3   # restrict on 3rd exact repeat

# Failing edit calls: after this many failures with identical args, inject guidance.
EDIT_FAIL_WARN = 2

# Agent turns allowed after the first successful edit before the run ends.
POST_EDIT_TURNS = 10

# Context-window management (prune older tool results to prevent token bloat).
KEEP_FULL_VIEWS = 3
KEEP_FULL_OTHER = 3
STALE_OUTPUT_CHARS = 1000

# Tools that are ALWAYS executed and never blocked/deduplicated.
EDIT_TOOLS = {"replace_lines"}
# Read-only tools subject to repeat-tracking.
READ_TOOLS = {"view_file", "search_code", "repo_map", "explore_directory",
              "run_python_repro", "run_bash", "run_pytest"}

DUP_PREFIX = "DUPLICATE CALL REJECTED"

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


class SWEBenchState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    max_iterations: int
    patch: str
    audit_passed: bool
    audit_count: int


SYSTEM_PROMPT = """You are an autonomous software engineer fixing a bug in an open-source Python repository.

Rules:
1. Use view_file to read code. Always view at least 60 lines at a time.
   - When supporting a parameter or fixing a class, inspect the entire class and related methods (e.g., both reading and writing, parsing and formatting), not just __init__.
2. Use run_python_repro to reproduce the bug and verify fixes.
   - The reproduction script MUST assert functional correctness (e.g., verify actual output values or round-trips, not just that an exception was avoided) and print PASS or FAIL.
3. Once you locate the faulty line, call replace_lines IMMEDIATELY.
   replace_lines(file_path, start_line, end_line, new_str) replaces lines
   start_line..end_line (1-indexed, inclusive). new_str must have correct indentation
   and must NOT include the lines just outside that range.
4. After a successful edit, run your repro to verify. You can also use run_bash to run the relevant existing pytest file to ensure no regressions. PASS → done. FAIL → fix the edit.
5. Do NOT keep re-reading lines you have already seen. Locate the bug, fix it completely, and verify.
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


def _last_repro_result(messages: list[BaseMessage]) -> str | None:
    """Return 'PASS', 'FAIL', or None if no repro has run yet."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == "run_python_repro":
            content = str(m.content)
            if content.startswith(DUP_PREFIX):
                continue
            if "PASS" in content and "FAIL" not in content and "AssertionError" not in content:
                return "PASS"
            if "FAIL" in content or "AssertionError" in content or "Traceback" in content:
                return "FAIL"
            return "UNKNOWN"
    return None


def _edits_since_last_repro(messages: list[BaseMessage]) -> int:
    """Count successful edits applied after the most recent valid run_python_repro."""
    edits = 0
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == "run_python_repro" and not str(m.content).startswith(DUP_PREFIX):
            break
        if _is_successful_edit(m):
            edits += 1
    return edits


def _last_tool_was_successful_edit(messages: list[BaseMessage]) -> bool:
    """Check if the most recent tool execution included a successful edit."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            return _is_successful_edit(m)
        if isinstance(m, AIMessage):
            break
    return False


def _has_run_repro(messages: list[BaseMessage]) -> bool:
    """Check if run_python_repro has been executed at least once."""
    return any(
        isinstance(m, ToolMessage)
        and m.name == "run_python_repro"
        and not str(m.content).startswith(DUP_PREFIX)
        for m in messages
    )


def _last_tool_was_repro_pass(messages: list[BaseMessage]) -> bool:
    """Check if the most recent tool executed was a passing run_python_repro."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            if m.name == "run_python_repro":
                content = str(m.content)
                return "PASS" in content and "FAIL" not in content and "AssertionError" not in content
            return False
        if isinstance(m, AIMessage):
            break
    return False


def _modified_files_from_diff(diff_text: str) -> list[str]:
    """Extract list of modified file paths relative to ROOT from git diff output."""
    files = []
    for line in diff_text.splitlines():
        if line.startswith("diff --git a/"):
            parts = line.split()
            if len(parts) >= 3 and parts[2].startswith("a/"):
                files.append(parts[2][2:])
    return files


def _find_associated_test_file(rel_path: str) -> str | None:
    """Find the likely test file for a modified source file."""
    dirname = posixpath.dirname(rel_path)
    basename = posixpath.basename(rel_path)
    if not basename.endswith(".py"):
        return None
    stem = basename[:-3]

    candidates = [
        posixpath.join(dirname, "tests", f"test_{stem}.py"),
        posixpath.join(dirname, "test", f"test_{stem}.py"),
        posixpath.join(dirname, f"test_{stem}.py"),
        posixpath.join("tests", dirname, f"test_{stem}.py"),
        posixpath.join("tests", f"test_{stem}.py"),
    ]
    for c in candidates:
        if ct.exists(c, "f"):
            return c
    return None


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

    last_repro = _last_repro_result(state["messages"])
    edits_since_repro = _edits_since_last_repro(state["messages"])
    reads_since_repro = sum(counts.values())

    # Force edit mode if:
    # 1. No edits and reached FORCE_EDIT_AFTER turns.
    # 2. Or any read tool was repeated >= VIEW_REPEAT_RESTRICT times.
    # 3. Or a test failed, 0 edits made since, and the agent has read 3+ times.
    force_edit = (
        (no_edit and state["iteration"] >= FORCE_EDIT_AFTER)
        or max_reps >= VIEW_REPEAT_RESTRICT
        or (last_repro == "FAIL" and edits_since_repro == 0 and reads_since_repro >= 3)
    )

    if force_edit:
        last_view = _last_view_content(state["messages"])
        ctx = f"\nFor reference, the last code you viewed:\n{last_view}\n" if last_view else ""
        msg = (
            "You have spent enough turns reading code or running redundant queries."
            f"{ctx}"
            "ONLY `replace_lines` is available now. Apply your fix immediately using "
            "the line numbers shown above."
        )
        messages = messages + [HumanMessage(content=msg)]
        model = llm_edit_only
    else:
        model = llm
        if _last_tool_was_successful_edit(state["messages"]):
            messages = messages + [
                HumanMessage(
                    content="Your edit has been applied. Verify with `run_python_repro` or `run_pytest`. "
                    "Make sure your test checks functional correctness (values and behavior, not just absence of errors)."
                )
            ]
        elif _last_tool_was_repro_pass(state["messages"]):
            messages = messages + [
                HumanMessage(
                    content="Your reproduction script returned PASS. However, a passing snippet alone does NOT guarantee the task is complete.\n"
                    "1. Check if other methods in the class/module need updating (e.g., both reading and writing, parsing and formatting).\n"
                    "2. Run the repository test suite with `run_pytest` to ensure no regressions.\n"
                    "3. If all tests pass and the implementation is complete, stop calling tools to conclude."
                )
            ]
        elif last_repro == "FAIL" and edits_since_repro == 0:
            messages = messages + [
                HumanMessage(
                    content="The previous verification test failed. If the failure was caused by a bug in the code, "
                    "call `replace_lines` to fix it. If your reproduction script had a mistake or needs additional assertions, "
                    "you may update and re-run `run_python_repro`."
                )
            ]

    response = model.invoke(messages)
    return {"messages": [response], "iteration": state["iteration"] + 1}


def tools_node(state: SWEBenchState) -> dict:
    messages = state["messages"]
    last     = messages[-1]

    read_counts = _read_call_counts(messages[:-1])
    fail_counts = _failed_edit_counts(messages[:-1])
    has_tested  = _has_run_repro(messages[:-1])
    edits_since_repro = _edits_since_last_repro(messages[:-1])

    results = []
    for call in last.tool_calls:
        name = call["name"]
        tool = TOOL_MAP.get(name)

        # GUARD 1: Prevent running the exact same test script without any code modifications
        if name == "run_python_repro":
            sig = _call_sig(call)
            n_repro = read_counts.get(sig, 0)
            if n_repro >= 1 and edits_since_repro == 0:
                content = (
                    f"{DUP_PREFIX}: This EXACT reproduction script was already executed and failed, and no files "
                    "have been modified since. Re-running the identical test will produce the exact same failure. "
                    "Modify the code with `replace_lines` to fix the bug, or update your test script."
                )
                results.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))
                read_counts[sig] = n_repro + 1
                continue

        # GUARD 2: Prevent duplicate identical read-only calls (e.g. view_file 3x with identical args)
        if name in READ_TOOLS and name != "run_python_repro":
            sig = _call_sig(call)
            n   = read_counts.get(sig, 0)
            if n >= VIEW_REPEAT_WARN + 1:
                content = (
                    f"{DUP_PREFIX}: You have already called `{name}` with these exact arguments {n + 1} times. "
                    "The output has not changed. Do not repeat identical queries. "
                    "Proceed with `replace_lines` to fix the bug, or view a different part of the codebase."
                )
                results.append(ToolMessage(content=content, tool_call_id=call["id"], name=name))
                read_counts[sig] = n + 1
                continue

        try:
            content = str(tool.invoke(call["args"])) if tool else f"Unknown tool {name}"
        except Exception as e_:
            content = f"Tool error: {e_}"

        if name in READ_TOOLS:
            sig = _call_sig(call)
            n   = read_counts.get(sig, 0)
            if n == VIEW_REPEAT_WARN:
                content += (
                    f"\n\n[WARNING: This is the 2nd time you called `{name}` with these exact arguments. "
                    "Do not repeat identical queries. Formulate your fix and call `replace_lines`.]"
                )
            read_counts[sig] = n + 1

        elif name in EDIT_TOOLS:
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


def audit_node(state: SWEBenchState) -> dict:
    diff = git_diff().strip()
    audit_count = state.get("audit_count", 0) + 1

    if not diff:
        return {
            "messages": [
                HumanMessage(
                    content="You have not modified any files yet. Apply your fix using `replace_lines`."
                )
            ],
            "audit_passed": False,
            "audit_count": audit_count,
        }

    modified_files = _modified_files_from_diff(diff)
    test_files_found = []
    for f in modified_files:
        tf = _find_associated_test_file(f)
        if tf and tf not in test_files_found:
            test_files_found.append(tf)

    # 1. Run unit test file(s) if found in the repository
    if test_files_found:
        test_target = " ".join(test_files_found)
        rc, out, err = ct.run_script(f"pytest -q {test_target}", timeout=90)
        combined = (out + "\n" + err).strip()

        if rc != 0 or "FAILED" in combined or "ERROR" in combined:
            tail = "\n".join(combined.splitlines()[-35:])
            msg = (
                f"[SYSTEM AUDIT FAILED]: Automated verification ran `{test_target}` and detected failures:\n"
                f"```\n{tail}\n```\n"
                "Your proposed changes broke existing tests or did not fully solve the issue.\n"
                "Do NOT finish yet. Use `view_file` to inspect the failure, correct the code with `replace_lines`, "
                "and verify with `run_pytest`."
            )
            return {
                "messages": [HumanMessage(content=msg)],
                "audit_passed": False,
                "audit_count": audit_count,
            }

    # 2. Check if reproduction script was run and passed
    last_repro = _last_repro_result(state["messages"])
    if last_repro != "PASS":
        msg = (
            "[SYSTEM AUDIT]: Repository unit tests passed, but you have not demonstrated that your reproduction script "
            "passes (PASS). Run `run_python_repro` with a functional test asserting the expected behavior before completing."
        )
        return {
            "messages": [HumanMessage(content=msg)],
            "audit_passed": False,
            "audit_count": audit_count,
        }

    # 3. Everything verified!
    return {
        "messages": [
            HumanMessage(
                content="[SYSTEM AUDIT PASSED]: Both reproduction and repository unit tests passed cleanly. Extracting final patch."
            )
        ],
        "audit_passed": True,
        "audit_count": audit_count,
    }


def route_decision(state: SWEBenchState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return "extract_patch"

    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"

    # Agent sent text without tool calls (thinks it is done).
    diff = git_diff().strip()
    if not diff and state["iteration"] < state["max_iterations"] - 2:
        return "nudge"

    # If audit has already verified and approved, extract patch!
    if state.get("audit_passed", False):
        return "extract_patch"

    # If maximum audits reached or max turns near, extract patch
    if state.get("audit_count", 0) >= 3 or state["iteration"] >= state["max_iterations"] - 2:
        return "extract_patch"

    # Otherwise, perform automated audit before accepting completion
    return "audit"


def route_after_audit(state: SWEBenchState) -> str:
    if state.get("audit_passed", False):
        return "extract_patch"
    if state.get("audit_count", 0) >= 3 or state["iteration"] >= state["max_iterations"] - 1:
        return "extract_patch"
    return "agent"


def extract_patch_node(state: SWEBenchState) -> dict:
    return {"patch": git_diff()}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
builder = StateGraph(SWEBenchState)
builder.add_node("agent",         agent_node)
builder.add_node("tools",         tools_node)
builder.add_node("nudge",         nudge_node)
builder.add_node("audit",         audit_node)
builder.add_node("extract_patch", extract_patch_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_decision,
    {"tools": "tools", "nudge": "nudge", "audit": "audit", "extract_patch": "extract_patch"},
)
builder.add_edge("tools",         "agent")
builder.add_edge("nudge",         "agent")
builder.add_conditional_edges(
    "audit",
    route_after_audit,
    {"agent": "agent", "extract_patch": "extract_patch"},
)
builder.add_edge("extract_patch", END)

swe_agent = builder.compile()