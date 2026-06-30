"""
Download Qwen3-VL-4B-Instruct from HuggingFace.
(v1.3.0 version, handles gated models with HF_TOKEN)
Supports: secrets.HF_TOKEN, workflow input hf_token, or HF_TOKEN env
"""
import os, sys, time
from huggingface_hub import snapshot_download, login

repo_id = os.environ.get('REPO_ID', 'Qwen/Qwen3-VL-4B-Instruct')
max_retries = 3

# Token priority: secrets.HF_TOKEN > HF_TOKEN_INPUT > HF_TOKEN env
hf_token = os.environ.get('HF_TOKEN', '') or os.environ.get('HF_TOKEN_INPUT', '')

if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        print(f"Logged in with HF_TOKEN (len={len(hf_token)})")
    except Exception as e:
        print(f"Login attempted (non-fatal): {e}")
else:
    print("WARNING: No HF_TOKEN found. Trying anonymous download...")
    print("(Qwen3-VL-4B-Instruct is gated; this may fail)")

for attempt in range(1, max_retries + 1):
    print(f"Downloading {repo_id}... (attempt {attempt}/{max_retries})")
    try:
        model_dir = snapshot_download(
            repo_id,
            local_dir='./model',
            token=hf_token if hf_token else None,
            resume_download=True,
            ignore_patterns=["*.pt", "*.bin"],
        )
        print(f"Downloaded to: {model_dir}")

        files = os.listdir(model_dir)
        print(f"Files: {len(files)} items")
        total = sum(os.path.getsize(os.path.join(model_dir, f)) for f in files if os.path.isfile(os.path.join(model_dir, f)))
        print(f"Total size: {total / 1e9:.2f} GB")

        with open(os.environ["GITHUB_ENV"], "a") as f:
            f.write(f"MODEL_PATH={model_dir}\n")
        break
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        if attempt == max_retries:
            print(f"\n== HOW TO GET HF_TOKEN ==")
            print(f"1. Open https://huggingface.co/settings/tokens")
            print(f"   (If blocked in China, use VPN or phone hotspot)")
            print(f"2. Click 'New token' -> name: github-actions -> role: read")
            print(f"3. Copy the token (starts with hf_)")
            print(f"4. Then trigger this workflow again with the token:")
            print(f"   GitHub -> Actions -> Qwen3-VL-4B -> Run workflow")
            print(f"   Paste token in 'HuggingFace token' field")
            print(f"")
            print(f"Or set it permanently as repo secret:")
            print(f"   Settings -> Secrets and variables -> Actions")
            print(f"   New secret -> Name: HF_TOKEN -> Value: <your-token>")
            sys.exit(1)
        print("Waiting 10s before retry...")
        time.sleep(10)
