# runner.py
import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset
from langchain_core.messages import HumanMessage, SystemMessage

from container import container_name, git_diff, prepare
from tools import EDIT_OK_PREFIX
from agent_graph import swe_agent, SYSTEM_PROMPT, DUP_PREFIX

MAX_ITER = int(os.getenv("MAX_ITERATIONS", "30"))
# This copy of SWE-bench Lite carries the `image` column (the princeton-nlp copy does not).
DATASET_ID = os.getenv("SWE_DATASET", "SWE-bench/SWE-bench_Lite")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SWE-bench agent on a given instance.")
    parser.add_argument("--instance_id", type=str, default="astropy__astropy-12907", help="SWE-bench instance ID")
    return parser.parse_args()


def calculate_patch_stats(patch_str: str) -> dict:
    if not patch_str.strip():
        return {"files": 0, "additions": 0, "deletions": 0}
    files, additions, deletions = 0, 0, 0
    for line in patch_str.splitlines():
        if line.startswith("diff --git"):
            files += 1
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return {"files": files, "additions": additions, "deletions": deletions}


def main():
    args = parse_args()
    instance_id = args.instance_id
    start_time = time.perf_counter()

    container_name()  # fail fast if run.ps1 has not started the container

    print(f"[*] Fetching metadata for instance: {instance_id} from {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID, split="test")
    matches = [inst for inst in dataset if inst["instance_id"] == instance_id]
    if not matches:
        raise ValueError(f"Instance '{instance_id}' not found in SWE-bench_Lite test split.")
    instance = matches[0]

    prepare(instance["base_commit"])

    metrics = {
        "instance_id": instance_id,
        "model_id": os.getenv("MODEL_ID", "z-ai/glm-5.2"),
        "total_turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls": Counter(),
        "edits_ok": 0,
        "edits_failed": 0,
        "duplicates_rejected": 0,
    }

    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Issue Description:\n{instance['problem_statement']}"),
        ],
        "iteration": 0,
        "max_iterations": MAX_ITER,
        "patch": "",
    }

    print(f"\n{'='*70}\n[+] Launching Agent: {instance_id} | Model: {metrics['model_id']}\n{'='*70}\n")

    final_patch = ""
    pending_edit_ids = set()
    interrupted = ""
    try:
      for event in swe_agent.stream(
        initial_state,
        config={"recursion_limit": MAX_ITER * 3 + 10},
        stream_mode="updates",
      ):
          for node_name, node_output in event.items():
              print(f"\n--- [Node: {node_name}] ---")
              if node_name == "agent":
                  metrics["total_turns"] += 1
                  msg = node_output["messages"][-1]
                  if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                      metrics["input_tokens"] += msg.usage_metadata.get("input_tokens", 0)
                      metrics["output_tokens"] += msg.usage_metadata.get("output_tokens", 0)
                  if msg.tool_calls:
                      for call in msg.tool_calls:
                          name = call["name"]
                          metrics["tool_calls"][name] += 1
                          if name == "edit_file":
                              pending_edit_ids.add(call["id"])
                          print(f"-> Action: {name}({call['args']})")
                  else:
                      print(f"-> Agent Response:\n{msg.content}")

              elif node_name == "tools":
                  for msg in node_output["messages"]:
                      text = str(msg.content)
                      if text.startswith(DUP_PREFIX):
                          metrics["duplicates_rejected"] += 1
                      if msg.tool_call_id in pending_edit_ids:
                          if text.startswith(EDIT_OK_PREFIX):
                              metrics["edits_ok"] += 1
                          else:
                              metrics["edits_failed"] += 1
                      preview = text[:200] + "..." if len(text) > 200 else text
                      print(f"-> Tool Output:\n{preview}")

              elif node_name == "extract_patch":
                  final_patch = node_output.get("patch", "")
    except BaseException as exc:  # includes KeyboardInterrupt and LLM timeouts
        interrupted = f"{type(exc).__name__}: {exc}"
        print(f"\n[!] Run interrupted: {interrupted}")
        try:
            final_patch = git_diff()
        except Exception as diff_exc:
            print(f"[!] Could not collect a patch after the interruption: {diff_exc}")
            final_patch = ""

    elapsed_time = time.perf_counter() - start_time
    patch_stats = calculate_patch_stats(final_patch)
    total_tokens = metrics["input_tokens"] + metrics["output_tokens"]

    if final_patch:
        prediction = {
            "model_name_or_path": metrics["model_id"],
            "instance_id": instance_id,
            "model_patch": final_patch,
        }
        with open("agent_prediction.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(prediction) + "\n")

        # Keep every run's diff on the host, since the container is thrown away after the run.
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patch_path = runs_dir / f"{instance_id}_{stamp}.patch"
        patch_path.write_text(final_patch, encoding="utf-8")
        print(f"[+] Patch saved to {patch_path}")

    print(f"\n{'='*70}")
    print(f"                  SWE-BENCH BENCHMARK METRICS REPORT")
    print(f"{'='*70}")
    print(f" Instance ID      : {instance_id}")
    print(f" Target Model     : {metrics['model_id']}")
    print(f" Status           : {'PATCH GENERATED' if final_patch else 'NO DIFF GENERATED'}")
    if interrupted:
        print(f" Interrupted      : {interrupted}")
    print(f" Wall Clock Time  : {elapsed_time:.2f}s")
    print(f" Total LLM Turns  : {metrics['total_turns']} / {MAX_ITER}")
    print(f" Total Tokens     : {total_tokens:,} (In: {metrics['input_tokens']:,} | Out: {metrics['output_tokens']:,})")
    print(f" Edits            : {metrics['edits_ok']} applied, {metrics['edits_failed']} failed")
    print(f" Duplicates       : {metrics['duplicates_rejected']} rejected")
    print(f" Tool Calls       : {dict(metrics['tool_calls'])}")
    print(f" Patch Stats      : {patch_stats['files']} files, +{patch_stats['additions']}, -{patch_stats['deletions']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()