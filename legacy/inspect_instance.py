# inspect_instance.py
from datasets import load_dataset
import json

# Load just the test split (SWE-bench Lite test set contains 300 instances)
swebench_lite  = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

# Pick the first instance
sample = swebench_lite[0]

print(f"Instance ID:    {sample['instance_id']}")
print(f"Repository:     {sample['repo']}")
print(f"Base Commit:    {sample['base_commit']}")
print(f"FAIL_TO_PASS:   {sample['FAIL_TO_PASS']}")
print(f"PASS_TO_PASS:   {sample['PASS_TO_PASS']}")
print("\n--- PROBLEM STATEMENT (Issue description given to the agent) ---\n")
print(sample["problem_statement"][:500] + "...\n")

# Save one complete instance to JSON to examine the full payload
with open("sample_instance.json", "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=2)
print("Saved complete instance to sample_instance.json")