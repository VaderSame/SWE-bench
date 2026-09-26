import json
import subprocess
from datasets import load_dataset

def fix_and_run_modal():
    print("1. Downloading SWE-bench dataset...")
    # Load the standard dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    dataset_list = []
    print("2. Injecting missing 'image' metadata for Modal...")
    for instance in dataset:
        # Modal needs the exact Docker Hub image name for the evaluation sandbox
        env_commit = instance.get("environment_setup_commit", "")
        if "image" not in instance:
            instance["image"] = f"swebench/sweb.env.x86_64.{env_commit}"
        dataset_list.append(instance)
        
    local_file = "swebench_lite_modal_fixed.json"
    with open(local_file, "w") as f:
        json.dump(dataset_list, f)
        
    print(f"3. Saved fixed dataset to {local_file}")
    print("4. Launching SWE-bench Modal evaluation...\n")
    
    # Run the SWE-bench evaluation using the fixed local dataset
    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", local_file,
        "--split", "test",
        "--predictions_path", "agent_prediction.jsonl",
        "--run_id", "modal_lite_eval",
        "--modal", "true"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    fix_and_run_modal()