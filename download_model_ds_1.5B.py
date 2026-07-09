"""
Download DeepSeek-R1-Distill-Qwen-1.5B from HuggingFace.
GitHub runners are in Azure US, HF access is fast.
Retries up to 3 times on failure.
"""
import os
import time
from huggingface_hub import snapshot_download, login

repo_id = os.environ.get("REPO_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
local_dir = f"./{repo_id.replace('/', '-')}"
max_retries = 3

# Try to login if token provided
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("Logged in to HuggingFace")
    except Exception as e:
        print(f"Login skipped (non-critical): {e}")

for attempt in range(1, max_retries + 1):
    print(f"Downloading {repo_id} to {local_dir}... (attempt {attempt}/{max_retries})")
    try:
        model_dir = snapshot_download(
            repo_id,
            local_dir=local_dir,
            resume_download=True,
            ignore_patterns=["*.safetensors.index.json", "*.md", "*.pdf"],
        )
        print(f"Downloaded to: {model_dir}")

        # List downloaded files
        for f in sorted(os.listdir(model_dir)):
            fpath = os.path.join(model_dir, f)
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                print(f"  {f}: {size_mb:.1f} MB")

        # Set MODEL_PATH for GitHub Actions
        github_env = os.environ.get("GITHUB_ENV", "")
        if github_env:
            with open(github_env, "a") as f:
                f.write(f"MODEL_PATH={model_dir}\n")
                print(f"Set MODEL_PATH={model_dir}")
        break
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        if attempt == max_retries:
            raise
        print("Waiting 10s before retry...")
        time.sleep(10)
