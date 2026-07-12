"""
Convert Qwen3-4B-Instruct-2507 to RKLLM format for RK3588 NPU.
16K context using patched RKLLM toolkit v1.2.2.
opt-level controlled by OPTIMIZATION_LEVEL env var (default 1).
"""
from rkllm.api import RKLLM
import os, sys

modelpath = os.environ.get('MODEL_PATH')
if not modelpath:
    print('MODEL_PATH environment variable is required')
    sys.exit(1)

max_context = int(os.environ.get('MAX_CONTEXT', '16384'))
if max_context < 32:
    max_context = 32

optimization_level = int(os.environ.get('OPTIMIZATION_LEVEL', '1'))

print(f'Loading model from: {modelpath}')
print(f'Max context: {max_context}')
print(f'Optimization level: {optimization_level}')
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
target_platform = "RK3588"
quantized_dtype = "W8A8"
quantized_algorithm = "normal"
num_npu_core = 3

print(f'Starting quantization: max_context={max_context}, dtype={quantized_dtype}, opt={optimization_level}...')
sys.stdout.flush()

try:
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
except Exception as e:
    print(f'Build raised exception: {e}')
    try:
        log = llm.get_log()
        print(f'LLM log: {log}')
    except:
        pass
    sys.exit(1)

if ret != 0:
    print(f'Build model failed! ret={ret}')
    try:
        output = llm.get_log()
        print(f'LLM log:\n{output}')
    except Exception as e:
        print(f'Could not get log: {e}')
    sys.exit(ret)

out_name = (
    f"Qwen3-4B-Instruct-2507-rk3588-{quantized_dtype.lower()}_g128"
    f"-opt-{optimization_level}-hybrid-ratio-0.0-{max_context // 1024}k.rkllm"
)
ret = llm.export_rkllm(f"./{out_name}")
if ret != 0:
    print('Export model failed!')
    sys.exit(ret)

print(f'Export success: {out_name}')
