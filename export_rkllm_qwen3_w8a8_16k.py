"""
Qwen3-4B → RKLLM — W8A8 only, v1.3.0, 16K context.
Skip W4A16 attempts entirely — fast path for baseline comparison.
"""
from rkllm.api import RKLLM
import os, sys, json

modelpath = os.environ.get('MODEL_PATH')
if not modelpath:
    print('MODEL_PATH required')
    sys.exit(1)

max_context = int(os.environ.get('MAX_CONTEXT', '16384'))
if max_context > 16384:
    max_context = 16384

print(f'Loading: {modelpath}')
print(f'Max context: {max_context}')

llm = RKLLM()
ret = llm.load_huggingface(model=modelpath, device='cpu', dtype='float16', load_weight=True)
if ret != 0:
    print(f'Load failed! ret={ret}')
    sys.exit(ret)
print('Model loaded')

print(f'Quantizing: w8a8 normal + dataset + target_platform (16K)...')
ret = llm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype="w8a8",
    quantized_algorithm="normal",
    dataset="./data_quant.json",
    target_platform="RK3588",
    num_npu_core=3,
    hybrid_rate=0.0,
    max_context=max_context,
)
if ret != 0:
    print(f'Quantization failed! ret={ret}')
    sys.exit(ret)

out_name = f"Qwen3-4B-Instruct-2507-rk3588-W8A8-opt-1-normal-16k.rkllm"
ret = llm.export_rkllm(out_name)
if ret != 0:
    print(f'Export failed! ret={ret}')
    sys.exit(ret)

print(f'\n✅ {out_name}')
summary = {"model": out_name, "dtype": "w8a8", "toolkit": "v1.3.0",
           "max_context": max_context}
with open("export_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
