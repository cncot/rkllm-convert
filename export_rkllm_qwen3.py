"""
Convert Qwen3-4B-Instruct-2507 to RKLLM format for RK3588 NPU.
Download from ModelScope (HF mirror accessible in China).

Usage:
    MODEL_PATH=./Qwen3-4B-Instruct-2507 MAX_CONTEXT=32768 python3 export_rkllm_qwen3.py
"""
from rkllm.api import RKLLM
import os

modelpath = os.environ.get('MODEL_PATH')
if not modelpath:
    print('MODEL_PATH environment variable is required')
    exit(1)

max_context = int(os.environ.get('MAX_CONTEXT', '32768'))

print(f'Loading model from: {modelpath}')
print(f'Max context: {max_context}')

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
    exit(ret)

dataset = "./data_quant.json"
target_platform = "RK3588"
quantized_dtype = "W8A8"
quantized_algorithm = "normal"
num_npu_core = 3
optimization_level = 1

print(f'Starting quantization: max_context={max_context}, dtype={quantized_dtype}...')
ret = llm.build(
    do_quantization=True,
    optimization_level=optimization_level,
    quantized_dtype=quantized_dtype,
    quantized_algorithm=quantized_algorithm,
    target_platform=target_platform,
    num_npu_core=num_npu_core,
    dataset=dataset,
    hybrid_rate=0,
    max_context=max_context
)
if ret != 0:
    print('Build model failed!')
    exit(ret)

# Match user's naming convention: Qwen3-4B-Instruct-2507-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.0-16k.rkllm
out_name = (
    f"Qwen3-4B-Instruct-2507-rk3588-{quantized_dtype.lower()}_g128"
    f"-opt-{optimization_level}-hybrid-ratio-0.0-{max_context // 1024}k.rkllm"
)
ret = llm.export_rkllm(f"./{out_name}")
if ret != 0:
    print('Export model failed!')
    exit(ret)

print(f'Export success: {out_name}')
