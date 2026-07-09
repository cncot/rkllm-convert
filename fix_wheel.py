#!/usr/bin/env python3
"""Fix RKLLM toolkit wheel platform tag so pip accepts it on modern runners."""
import zipfile, os, shutil, sys

SRC = 'rkllm_toolkit-1.2.2-cp311-cp311-linux_x86_64_patched.whl'
DST = 'rkllm_toolkit-1.2.2-cp311-cp311-manylinux2014_x86_64.whl'

if not os.path.exists(SRC):
    print(f"ERROR: {SRC} not found")
    sys.exit(1)

print(f"Fixing wheel: {SRC} -> {DST}")

# Extract, patch WHEEL metadata, repack
TMPDIR = '/tmp/wheel_fix'
if os.path.exists(TMPDIR):
    shutil.rmtree(TMPDIR)
os.makedirs(TMPDIR)

with zipfile.ZipFile(SRC, 'r') as zin:
    for item in zin.namelist():
        content = zin.read(item)
        if item.endswith('WHEEL') or item.endswith('METADATA'):
            text = content.decode('utf-8')
            if 'linux_x86_64_patched' in text:
                text = text.replace('linux_x86_64_patched', 'manylinux2014_x86_64')
                print(f"  Patched {item}")
                content = text.encode('utf-8')
        out_path = os.path.join(TMPDIR, item)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(content)

with zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED) as zout:
    for root, dirs, files in os.walk(TMPDIR):
        for file in files:
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, TMPDIR)
            zout.write(filepath, arcname)

shutil.rmtree(TMPDIR)
os.remove(SRC)
print(f"Done: {DST} ({os.path.getsize(DST)} bytes)")
