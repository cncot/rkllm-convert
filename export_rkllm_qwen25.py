"""
Convert Qwen2.5-7B-Instruct to RKLLM format for RK3588 NPU.
Memory-optimized for GitHub Actions runner (14GB RAM).
"""
from rkllm.api import RKLLM
import os

modelpath = os.environ.get('MODEL_PATH', '/path/to/model')
max_context = int(os.environ.get('MAX_CONTEXT', '4096'))

llm = RKLLM()

# Use low_cpu_mem_usage for 7B model in limited RAM
ret = llm.load_huggingface(
    model=modelpath,
    model_lora=None,
    device='cpu',
    dtype='float16',
    custom_config={'low_cpu_mem_usage': True},
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

print(f'Starting quantization: max_context={max_context}...')
ret = llm.build(
    do_quantization=True,
    optimization_level=1,
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

out_name = f"{os.path.basename(modelpath)}_{quantized_dtype}_{target_platform}.rkllm"
ret = llm.export_rkllm(f"./{out_name}")
if ret != 0:
    print('Export model failed!')
    exit(ret)

print(f'Export success: {out_name}')
