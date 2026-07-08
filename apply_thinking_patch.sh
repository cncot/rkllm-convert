#!/bin/bash
# ============================================================
# Apply enable_thinking=True to OrangePi NPU inference server
# ============================================================
# Usage: bash apply_thinking_patch.sh
# This script patches the NPU server to enable thinking output.
# ============================================================

ORANGEPI="5plus@192.168.8.253"
SSH_PASS="jr@#75501252"

echo "=== Step 1: Enable thinking in NPU server ==="
sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no "$ORANGEPI" "

# Find the NPU server file
NPU_SERVER=\$(find /home/5plus/qwenpaw/ -name 'fastapi_server_llm.py' -type f 2>/dev/null | head -1)
if [ -z \"\$NPU_SERVER\" ]; then
    NPU_SERVER=\$(find /home/5plus/ -name 'fastapi_server_llm.py' -type f 2>/dev/null | head -1)
fi
echo 'NPU server: \$NPU_SERVER'

if [ -n \"\$NPU_SERVER\" ]; then
    # Change enable_thinking from False to True
    sed -i 's/enable_thinking = False/enable_thinking = True/g' \"\$NPU_SERVER\"
    echo '✅ Set enable_thinking = True'

    # Verify
    grep -n \"enable_thinking\" \"\$NPU_SERVER\"
else
    echo '⚠️  NPU server not found, check path'
fi
"

echo ""
echo "=== Step 2: Update model path to Qwen3-1.7B ==="
echo "⚠️  Manually update MODEL_PATH in fastapi_server_llm.py after deploying the .rkllm file"
echo ""
echo "=== Step 3: Restart NPU service ==="
echo "sshpass -p 'jr@#75501252' ssh 5plus@192.168.8.253 'sudo systemctl restart rkllm-npu.service'"
echo ""
echo "=== Done ==="
