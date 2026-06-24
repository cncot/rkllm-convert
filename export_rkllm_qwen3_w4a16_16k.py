"""
Convert Qwen3-4B-Instruct-2507 to RKLLM for RK3588 NPU — W4A16_G32 16K (INT4).
16K context = smaller KV cache = faster inference, lower IOVA pressure.
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
    print(f'Load model failed! ret={ret}')
    sys.exit(ret)

print('Model loaded successfully')
sys.stdout.flush()

dataset = "./data_quant.json"
num_npu_core = 3
target_platform = "RK3588"
quantized_dtype = "w4a16_g32"

print(f'Starting quantization: max_context={max_context}, dtype={quantized_dtype}...')
sys.stdout.flush()

# ── 依次尝试多种参数组合，确保 W4A16_G32 成功 ──
attempts = [
    # (alg, opt_level, with_dataset, with_target, desc)
    ("default",    1, True,  True,  "default + dataset + target_platform"),
    ("default",    1, True,  False, "default + dataset, no target"),
    ("default",    1, False, True,  "default, no dataset + target"),
    ("normal",     1, True,  True,  "normal + dataset + target_platform"),
    ("normal",     1, True,  False, "normal + dataset, no target"),
    ("normal",     1, False, True,  "normal, no dataset + target"),
]

success = False
final_ret = -1
for alg, opt_level, use_dataset, use_target, desc in attempts:
    kwargs = {
        "do_quantization": True,
        "optimization_level": opt_level,
        "quantized_dtype": quantized_dtype,
        "quantized_algorithm": alg,
        "num_npu_core": num_npu_core,
        "hybrid_rate": 0,
        "max_context": max_context,
    }
    if use_dataset:
        kwargs["dataset"] = dataset
    if use_target:
        kwargs["target_platform"] = target_platform

    print(f'Attempt: {desc}...')
    sys.stdout.flush()
    ret = llm.build(**kwargs)
    if ret == 0:
        print(f'  ✅ 成功! (alg={alg}, opt={opt_level}, dataset={use_dataset}, target={use_target})')
        sys.stdout.flush()
        success = True
        break
    else:
        print(f'  ❌ ret={ret}')
        sys.stdout.flush()
    final_ret = ret

if not success:
    print(f'\n❌ W4A16_G32 所有尝试均失败 (final ret={final_ret})')
    print('   RKLLM Toolkit v1.2.3 可能不支持此模型的 W4A16_G32 量化')
    sys.exit(final_ret)

# 构建输出文件名
out_name = (
    f"Qwen3-4B-Instruct-2507-rk3588-{quantized_dtype.lower()}"
    f"-opt-{opt_level}-hybrid-ratio-0.0-{max_context // 1024}k.rkllm"
)
print(f'Exporting to: {out_name}')
sys.stdout.flush()

ret = llm.export_rkllm(f"./{out_name}")
if ret != 0:
    print(f'Export model failed! ret={ret}')
    sys.exit(ret)

print(f'Export success: {out_name}')
