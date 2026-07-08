"""
Download Qwen3-1.7B-Instruct-2507 from HuggingFace.
GH runners are in Azure US, HF is much faster than cross-border ModelScope.
Retries up to 3 times on failure.
"""
import os
import time
from huggingface_hub import snapshot_download, login

# Check for HuggingFace token (set as secret in GitHub Actions)
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    print("Logged in to HuggingFace with provided token")

repo_id = os.environ.get("REPO_ID", "Qwen/Qwen3-1.7B-Instruct-2507")
local_dir = f"./{repo_id.replace('/', '-')}"
max_retries = 3

for attempt in range(1, max_retries + 1):
    print(f"Downloading {repo_id} to {local_dir}... (attempt {attempt}/{max_retries})")
    try:
        model_dir = snapshot_download(
            repo_id,
            local_dir=local_dir,
            resume_download=True,
            ignore_patterns=["*.safetensors.index.json"],  # Save space
        )
        print(f"Downloaded to: {model_dir}")
        # Set MODEL_PATH for next steps in GitHub Actions
        github_env = os.environ.get("GITHUB_ENV", "")
        if github_env:
            with open(github_env, "a") as f:
                f.write(f"MODEL_PATH={model_dir}\n")
        break
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        if attempt == max_retries:
            raise
        print("Waiting 10s before retry...")
        time.sleep(10)
