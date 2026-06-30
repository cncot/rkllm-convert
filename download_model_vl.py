"""
Download Qwen3-VL-4B-Instruct from HuggingFace.
(v1.3.0 version, handles gated models with HF_TOKEN)
"""
import os, sys, time
from huggingface_hub import snapshot_download, login

repo_id = os.environ.get('REPO_ID', 'Qwen/Qwen3-VL-4B-Instruct')
max_retries = 3

# Try to login with token if available
hf_token = os.environ.get('HF_TOKEN', '')
if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        print(f"Logged in with HF_TOKEN (len={len(hf_token)})")
    except Exception as e:
        print(f"Login attempted (non-fatal): {e}")

for attempt in range(1, max_retries + 1):
    print(f"Downloading {repo_id}... (attempt {attempt}/{max_retries})")
    try:
        model_dir = snapshot_download(
            repo_id,
            local_dir='./model',
            token=hf_token if hf_token else None,
            resume_download=True,
            ignore_patterns=["*.pt", "*.bin"],  # skip pytorch, keep safetensors
        )
        print(f"Downloaded to: {model_dir}")

        # Check what was downloaded
        files = os.listdir(model_dir)
        print(f"Files: {len(files)} items, total size: ", end="")
        total = sum(os.path.getsize(os.path.join(model_dir, f)) for f in files if os.path.isfile(os.path.join(model_dir, f)))
        print(f"{total / 1e9:.2f} GB")

        # Set MODEL_PATH for subsequent steps
        with open(os.environ["GITHUB_ENV"], "a") as f:
            f.write(f"MODEL_PATH={model_dir}\n")
        break
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        if attempt == max_retries:
            print(f"\nNOTE: Qwen3-VL-4B-Instruct is a GATED model.")
            print("You need to:")
            print("  1. Accept terms at: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct")
            print("  2. Get a token at: https://huggingface.co/settings/tokens")
            print("  3. Set it as repo secret: HF_TOKEN")
            print("     (Settings > Secrets and variables > Actions > New repository secret)")
            sys.exit(1)
        print("Waiting 10s before retry...")
        time.sleep(10)
