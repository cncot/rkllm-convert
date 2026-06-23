"""
Convert Qwen3-4B-Instruct-2507 to RKLLM for RK3588 NPU — W4A16 16K (INT4).
16K context = smaller KV cache = faster inference, lower IOVA pressure.
Falls back to W8A8 if W4A16 not supported by RKLLM v1.2.3.
"""
from rkllm.api import RKLLM
import os, sys

modelpath = os.environ.get('MODEL_PATH')
if not modelpath:
    print('MODEL_PATH environment variable is required')
    sys.exit(1)

max_context = int(os.environ.get('MAX_CONTEXT', '16384'))
if max_context > 16384:
    print(f'Clamping max_context from {max_context} to 16384 (16K target)')
    max_context = 16384
if max_context < 32:
    max_context = 32

print(f'Loading model from: {modelpath}')
print(f'Max context: {max_context}')
sys.stdout.flush()

llm = RKLLM()

ret = llm.load_huggingface(
    model=modelpath,
    model_lora=None,
    device='cpu',
    dtype='float16',
    custom_config=None,
    load_weight=True
)
if ret != 0:
    print('Load model failed!')
    sys.exit(ret)

print('Model loaded successfully')
sys.stdout.flush()

dataset = "./data_quant.json"
num_npu_core = 3
optimization_level = 1

# ---- 尝试 W4A16 ----
quantized_dtype = "w4a16_g32"
target_platform = "RK3588"

print(f'Starting quantization: max_context={max_context}, dtype={quantized_dtype}...')
print(f'Attempt 1: with target_platform={target_platform}')
sys.stdout.flush()

ret = llm.build(
    do_quantization=True,
    optimization_level=optimization_level,
    quantized_dtype=quantized_dtype,
    quantized_algorithm="normal",
    target_platform=target_platform,
    num_npu_core=num_npu_core,
    dataset=dataset,
    hybrid_rate=0,
    max_context=max_context
)

# Attempt 2: without target_platform
if ret != 0:
    print(f'Attempt 1 failed: ret={ret}. Trying W4A16 without target_platform...')
    sys.stdout.flush()
    ret = llm.build(
        do_quantization=True,
        optimization_level=optimization_level,
        quantized_dtype=quantized_dtype,
        quantized_algorithm="normal",
        num_npu_core=num_npu_core,
        dataset=dataset,
        hybrid_rate=0,
        max_context=max_context
    )

# Attempt 3: W8A8 fallback
if ret != 0:
    print(f'W4A16 not supported, falling back to W8A8...')
    sys.stdout.flush()
    quantized_dtype = "W8A8"
    ret = llm.build(
        do_quantization=True,
        optimization_level=optimization_level,
        quantized_dtype=quantized_dtype,
        quantized_algorithm="normal",
        target_platform=target_platform,
        num_npu_core=num_npu_core,
        dataset=dataset,
        hybrid_rate=0,
        max_context=max_context
    )

if ret != 0:
    print(f'All build attempts failed! ret={ret}')
    sys.exit(ret)

out_name = (
    f"Qwen3-4B-Instruct-2507-rk3588-{quantized_dtype.lower()}"
    f"-opt-{optimization_level}-hybrid-ratio-0.0-{max_context // 1024}k.rkllm"
)
ret = llm.export_rkllm(f"./{out_name}")
if ret != 0:
    print('Export model failed!')
    sys.exit(ret)

print(f'Export success: {out_name}')
