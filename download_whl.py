#!/usr/bin/env python3
"""
Download rkllm_toolkit patched whl from multiple sources.
Used by GitHub Actions workflow when the whl is not found locally.
"""
import requests, os, sys, time

WHL_NAME = "rkllm_toolkit-1.2.2-cp311-cp311-linux_x86_64_patched.whl"

sources = [
    # Source 1: GitHub raw content (public, may be rate-limited)
    {
        'url': 'https://raw.githubusercontent.com/cncot/rkllm-convert/main/' + WHL_NAME,
        'headers': {},
    },
    # Source 2: GitHub API (authenticated via GITHUB_TOKEN)
    {
        'url': 'https://api.github.com/repos/cncot/rkllm-convert/contents/' + WHL_NAME,
        'headers': {
            'Authorization': 'Bearer ' + os.environ.get('GH_TOKEN', ''),
            'Accept': 'application/vnd.github.v3.raw',
        },
    },
    # Source 3: jsDelivr CDN (GitHub mirror, no auth needed)
    {
        'url': 'https://cdn.jsdelivr.net/gh/cncot/rkllm-convert@main/' + WHL_NAME,
        'headers': {},
    },
]

for src in sources:
    url = src['url']
    headers = {k: v for k, v in src['headers'].items() if v}
    try:
        print('Trying: ' + url[:80] + '...')
        r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
        size = len(r.content)
        if r.status_code == 200 and size > 100000:
            with open(WHL_NAME, 'wb') as f:
                f.write(r.content)
            print('✅ Downloaded (' + str(size) + ' bytes)')
            break
        else:
            print('  status=' + str(r.status_code) + ', size=' + str(size))
    except Exception as e:
        print('  error: ' + str(e))
    time.sleep(1)
else:
    print()
    print('⚠️  Could not download patched whl automatically.')
    print('   Please manually get it from:')
    print('   https://github.com/cncot/rkllm-convert/blob/main/' + WHL_NAME)
    print('   Download the file and commit it to the repo root, then re-run.')
    sys.exit(1)
