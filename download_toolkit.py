#!/usr/bin/env python3
"""Download patched RKLLM Toolkit v1.2.2 whl from multiple sources."""
import requests, sys, os

urls = [
    'https://github.com/cncot/rkllm-convert/releases/download/toolkit-v1.2.2/rkllm_toolkit-1.2.2-cp311-cp311-linux_x86_64_patched.whl',
    'https://raw.githubusercontent.com/cncot/rkllm-convert/main/rkllm_toolkit-1.2.2-cp311-cp311-linux_x86_64_patched.whl',
    'https://github.com/cncot/rkllm-convert/raw/main/rkllm_toolkit-1.2.2-cp311-cp311-linux_x86_64_patched.whl',
]

output = 'rkllm_toolkit-1.2.2-cp311-cp311-linux_x86_64_patched.whl'

# Check if already exists
if os.path.exists(output):
    print(f"Found local {output}")
    sys.exit(0)

for url in urls:
    try:
        print(f"Downloading from: {url[:60]}...")
        r = requests.get(url, timeout=120, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 100000:
            with open(output, 'wb') as f:
                f.write(r.content)
            print(f"Downloaded: {len(r.content)} bytes")
            sys.exit(0)
    except Exception as e:
        print(f"Failed from {url}: {e}")

print("ERROR: Could not download patched whl from any source.")
sys.exit(1)
