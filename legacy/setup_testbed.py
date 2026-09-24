# setup_testbed.py
import json
import os
import subprocess
from pathlib import Path
from datasets import load_dataset

def run_git(cmd: str, cwd: Path) -> str:
    """Executes a git command and returns the stdout."""
    res = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        check=True,
        capture_output=True,
        text=True
    )
    return res.stdout.strip()

# 1. Load the first instance from SWE-bench Lite
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
instance = dataset[0]

repo_name = instance["repo"]              # e.g., "astropy/astropy"
base_commit = instance["base_commit"]      # Commit right before the bug was resolved
gold_patch = instance["patch"]            # The ground truth developer fix
instance_id = instance["instance_id"]

print(f"[*] Setting up instance: {instance_id}")
print(f"[*] Repository: {repo_name} @ {base_commit[:8]}")

# 2. Local workspace directory
workspace = Path("./testbeds") / instance_id
os.makedirs(workspace.parent, exist_ok=True)

# 3. Clone if not already present
if not (workspace / ".git").exists():
    print(f"[*] Cloning https://github.com/{repo_name}.git into {workspace}...")
    subprocess.run(
        f"git clone https://github.com/{repo_name}.git {workspace}",
        shell=True,
        check=True
    )

# 4. Clean and reset the repository to the base commit
print(f"[*] Hard resetting to base_commit: {base_commit}")
run_git("git clean -fdx", cwd=workspace)
run_git(f"git reset --hard {base_commit}", cwd=workspace)

# 5. Apply the gold patch to verify how SWE-bench captures patches
patch_file = workspace / "gold_patch.diff"
patch_file.write_text(gold_patch, encoding="utf-8")

print("[*] Applying gold patch to simulate a successful fix...")
run_git(f"git apply {patch_file.name}", cwd=workspace)

# 6. Extract the git diff (this is what your LangGraph agent must return)
generated_diff = run_git("git diff", cwd=workspace)
print("\n--- Extracted Model Patch (Sample Preview) ---")
print("\n".join(generated_diff.splitlines()[:15]))
print("...")

# 7. Package into SWE-bench prediction format
prediction = {
    "model_name_or_path": "poc-baseline",
    "instance_id": instance_id,
    "model_patch": generated_diff
}

with open("sample_prediction.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps(prediction) + "\n")

print("\n[+] Success! Saved valid benchmark prediction to sample_prediction.jsonl")