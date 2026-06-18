"""
Download Qwen3-4B-Instruct-2507 from ModelScope (HF mirror in China).
Saves to local directory and writes MODEL_PATH to GITHUB_ENV.
Retries up to 3 times on failure.
"""
import os
import time
from modelscope.hub.snapshot_download import snapshot_download

repo_id = os.environ['REPO_ID']
local_dir = f'./{repo_id.replace("/", "-")}'
max_retries = 3

for attempt in range(1, max_retries + 1):
    print(f"Downloading {repo_id} to {local_dir}... (attempt {attempt}/{max_retries})")
    try:
        model_dir = snapshot_download(repo_id, local_dir=local_dir)
        print(f"Downloaded to: {model_dir}")
        with open(os.environ["GITHUB_ENV"], "a") as f:
            f.write(f"MODEL_PATH={model_dir}\n")
        break
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        if attempt == max_retries:
            raise
        print("Waiting 10s before retry...")
        time.sleep(10)
