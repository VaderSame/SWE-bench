# runner.py
import argparse
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import structlog
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    """
    Configure structlog.
    Set LOG_FORMAT=json in .env or environment for newline-delimited JSON output.
    """
    use_json = os.getenv("LOG_FORMAT", "").lower() == "json"

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
    ]

    if use_json:
        processors = shared + [
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared + [
            structlog.dev.ConsoleRenderer(
                exception_formatter=structlog.dev.plain_traceback,
            )
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


_configure_logging()
log = structlog.get_logger("runner")


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from datasets import load_dataset
from langchain_core.messages import HumanMessage, SystemMessage

from container import container_name, git_diff, prepare
from tools import EDIT_OK_PREFIX
from agent_graph import swe_agent, SYSTEM_PROMPT, DUP_PREFIX, EDIT_TOOLS

MAX_ITER   = int(os.getenv("MAX_ITERATIONS", "30"))
DATASET_ID = os.getenv("SWE_DATASET", "SWE-bench/SWE-bench_Lite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWE-bench agent on a given instance.")
    parser.add_argument("--instance_id", type=str, default="astropy__astropy-12907")
    return parser.parse_args()


def calculate_patch_stats(patch_str: str) -> dict:
    if not patch_str.strip():
        return {"files": 0, "additions": 0, "deletions": 0}
    files = additions = deletions = 0
    for line in patch_str.splitlines():
        if line.startswith("diff --git"):
            files += 1
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return {"files": files, "additions": additions, "deletions": deletions}


def _action_fields(tool_name: str, args: dict) -> dict:
    """Extract parsed, intuitive metadata from tool call arguments."""
    if tool_name == "view_file":
        s = args.get("start_line", 1)
        e = args.get("end_line", 0) or (s + 149)
        return {"file": args.get("file_path", ""), "lines": f"{s}-{e}"}
    if tool_name == "replace_lines":
        new_str = args.get("new_str", "")
        s, e = args.get("start_line"), args.get("end_line")
        return {"file": args.get("file_path", ""), "lines": f"{s}-{e}", "new_lines": new_str.count("\n") + 1}
    if tool_name == "search_code":
        out = {"query": args.get("query", "")}
        if d := args.get("directory"): out["in"] = d
        return out
    if tool_name == "run_python_repro":
        return {"lines_of_code": args.get("code", "").count("\n") + 1}
    if tool_name == "run_bash":
        cmd = args.get("command", "")
        return {"cmd": (cmd[:50] + "…") if len(cmd) > 50 else cmd}
    if tool_name == "explore_directory":
        return {"path": args.get("path", ".")}
    if tool_name == "repo_map":
        return {"dir": args.get("target_directory", ".")}
    return {k: str(v)[:50] for k, v in args.items()}


def _parsed_metadata(tool_name: str, content: str) -> tuple[str, dict]:
    """Extract parsed, intuitive metadata from a tool's output string."""
    if tool_name == "run_python_repro":
        if "PASS" in content: return "info", {"verdict": "PASS"}
        if "FAIL" in content: return "warning", {"verdict": "FAIL"}
        return "error", {"verdict": "ERROR"}
    if tool_name == "replace_lines":
        if content.startswith(EDIT_OK_PREFIX): return "info", {"status": "applied"}
        return "warning", {"status": "failed"}
    if tool_name == "view_file":
        if "Error" in content[:20] or "File has" in content[:20]: return "warning", {"status": "error"}
        lines_shown = content.count("|\n") + (1 if content.endswith("|") else 0) or content.count("\n")
        return "info", {"lines_shown": lines_shown}
    if tool_name == "search_code":
        matches = [l for l in content.splitlines() if l.strip() and not l.lower().startswith("no ")]
        return "info", {"matches": len(matches)}
    if content.startswith(DUP_PREFIX):
        return "warning", {"status": "duplicate"}
    
    return "info", {"length": len(content)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args        = parse_args()
    instance_id = args.instance_id
    start_time  = time.perf_counter()

    structlog.contextvars.bind_contextvars(instance_id=instance_id)

    container_name()   # fail-fast if container not running

    log.info("fetch_metadata", dataset=DATASET_ID)
    dataset = load_dataset(DATASET_ID, split="test")
    matches = [inst for inst in dataset if inst["instance_id"] == instance_id]
    if not matches:
        raise ValueError(f"Instance '{instance_id}' not found in {DATASET_ID}.")
    instance = matches[0]

    prepare(instance["base_commit"])

    model_id = os.getenv("MODEL_ID", "unknown")
    metrics  = {
        "instance_id":        instance_id,
        "model_id":           model_id,
        "total_turns":        0,
        "input_tokens":       0,
        "output_tokens":      0,
        "tool_calls":         Counter(),
        "edits_ok":           0,
        "edits_failed":       0,
        "duplicates_rejected": 0,
    }

    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Issue Description:\n{instance['problem_statement']}"),
        ],
        "iteration":      0,
        "max_iterations": MAX_ITER,
        "patch":          "",
    }

    log.info("agent_start", model=model_id, max_iter=MAX_ITER)

    final_patch: str = ""
    pending_edit_ids: set[str] = set()
    interrupted: str = ""

    try:
        for event in swe_agent.stream(
            initial_state,
            config={"recursion_limit": MAX_ITER * 3 + 10},
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():

                if node_name == "agent":
                    metrics["total_turns"] += 1
                    turn = metrics["total_turns"]
                    
                    # Restored Turn Header
                    print(f"\n{'─' * 12} Turn {turn}/{MAX_ITER} {'─' * 12}")
                    print("--- [Node: agent] ---")
                    
                    msg  = node_output["messages"][-1]

                    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        metrics["input_tokens"]  += msg.usage_metadata.get("input_tokens", 0)
                        metrics["output_tokens"] += msg.usage_metadata.get("output_tokens", 0)

                    if msg.tool_calls:
                        for call in msg.tool_calls:
                            name = call["name"]
                            metrics["tool_calls"][name] += 1
                            if name in EDIT_TOOLS:
                                pending_edit_ids.add(call["id"])

                            # Exact original action output
                            print(f"-> Action: {name}({call['args']})")
                            
                            # Parsed metadata (instead of raw args dict dump)
                            parsed_args = _action_fields(name, call["args"])
                            log.info("agent_action", tool=name, **parsed_args)
                    else:
                        content = msg.content or ""
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"-> Agent Text: {preview}")
                        log.info("agent_thinking", chars=len(content))

                elif node_name == "tools":
                    print("\n--- [Node: tools] ---")
                    for msg in node_output["messages"]:
                        text     = str(msg.content)
                        is_dup   = text.startswith(DUP_PREFIX)
                        is_edit  = msg.tool_call_id in pending_edit_ids

                        if is_dup:
                            metrics["duplicates_rejected"] += 1
                        if is_edit:
                            if text.startswith(EDIT_OK_PREFIX):
                                metrics["edits_ok"] += 1
                            else:
                                metrics["edits_failed"] += 1

                        # Exact original code output limit (truncates exactly like it did earlier)
                        preview = text[:300] + "..." if len(text) > 300 else text
                        print(f"-> Tool Output:\n{preview}")

                        # Parsed metadata for the result
                        level, fields = _parsed_metadata(msg.name, text)
                        getattr(log, level)("tool_result", tool=msg.name, **fields)

                elif node_name == "extract_patch":
                    final_patch = node_output.get("patch", "")

    except BaseException as exc:
        interrupted = f"{type(exc).__name__}: {exc}"
        log.error("run_interrupted", reason=interrupted, exc_info=True)
        try:
            final_patch = git_diff()
        except Exception as diff_exc:
            log.warning("patch_collect_failed", error=str(diff_exc))
            final_patch = ""

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    elapsed_time = time.perf_counter() - start_time
    patch_stats  = calculate_patch_stats(final_patch)
    total_tokens = metrics["input_tokens"] + metrics["output_tokens"]

    if final_patch:
        prediction = {
            "model_name_or_path": metrics["model_id"],
            "instance_id":        instance_id,
            "model_patch":        final_patch,
        }
        with open("agent_prediction.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction) + "\n")

        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        patch_path = runs_dir / f"{instance_id}_{stamp}.patch"
        patch_path.write_text(final_patch, encoding="utf-8")
        log.info("patch_saved", path=str(patch_path))

    # ------------------------------------------------------------------
    # Metrics report
    # ------------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"{'SWE-BENCH BENCHMARK METRICS REPORT':^70}")
    print(sep)
    print(f"  Instance ID      : {instance_id}")
    print(f"  Target Model     : {metrics['model_id']}")
    print(f"  Status           : {'PATCH GENERATED' if final_patch else 'NO DIFF GENERATED'}")
    if interrupted:
        print(f"  Interrupted      : {interrupted}")
    print(f"  Wall Clock Time  : {elapsed_time:.2f}s")
    print(f"  Total LLM Turns  : {metrics['total_turns']} / {MAX_ITER}")
    print(f"  Total Tokens     : {total_tokens:,}  (In: {metrics['input_tokens']:,} | Out: {metrics['output_tokens']:,})")
    print(f"  Edits            : {metrics['edits_ok']} applied, {metrics['edits_failed']} failed")
    print(f"  Duplicates       : {metrics['duplicates_rejected']} rejected")
    print(f"  Tool Calls       : {dict(metrics['tool_calls'])}")
    print(f"  Patch Stats      : {patch_stats['files']} files, +{patch_stats['additions']}, -{patch_stats['deletions']}")
    print(f"{sep}\n")

    log.info(
        "run_complete",
        status="patch_generated" if final_patch else "no_diff",
        wall_time_s=round(elapsed_time, 2),
        turns=metrics["total_turns"],
        total_tokens=total_tokens,
        edits_ok=metrics["edits_ok"],
        edits_failed=metrics["edits_failed"],
        patch_files=patch_stats["files"],
        patch_add=patch_stats["additions"],
        patch_del=patch_stats["deletions"],
    )

if __name__ == "__main__":
    main()