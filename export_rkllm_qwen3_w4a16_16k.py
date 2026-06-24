"""
Convert Qwen3-4B-Instruct-2507 to RKLLM for RK3588 NPU.
Strategy: Try W4A16_G32 -> W4A16 -> W8A8 (all 16K).
v1.2.3 doesn't support W4A16 on RK3588, so falls back to W8A8.
"""
from rkllm.api import RKLLM
import os, sys, json

modelpath = os.environ.get('MODEL_PATH')
if not modelpath:
    print('MODEL_PATH environment variable is required')
    sys.exit(1)

max_context = int(os.environ.get('MAX_CONTEXT', '16384'))
if max_context > 16384:
    max_context = 16384
if max_context < 32:
    max_context = 32

print(f'Loading model from: {modelpath}')
print(f'Max context: {max_context}')

llm = RKLLM()
ret = llm.load_huggingface(
    model=modelpath, model_lora=None, device='cpu',
    dtype='float16', custom_config=None, load_weight=True
)
if ret != 0:
    print(f'Load model failed! ret={ret}')
    sys.exit(ret)
print('Model loaded successfully')

dataset = "./data_quant.json"
num_npu_core = 3
target_platform = "RK3588"

# ── 策略：先试 W4A16_G32 → W4A16 → W8A8 ──
dtype_attempts = [
    ("w4a16_g32", ["gdq", "normal"]),
    ("w4a16",     ["gdq", "normal"]),
    ("w8a8",      ["normal", "gdq"]),
]

success = False
used_dtype = used_alg = used_target = used_dataset = None

for quantized_dtype, algorithms in dtype_attempts:
    if success:
        break
    for alg in algorithms:
        for use_dataset, use_target, desc_suffix in [
            (True, True, "dataset+target"),
            (True, False, "dataset only"),
            (False, True, "target only"),
            (False, False, "no dataset, no target"),
        ]:
            kwargs = {
                "do_quantization": True,
                "optimization_level": 1,
                "quantized_dtype": quantized_dtype,
                "quantized_algorithm": alg,
                "num_npu_core": num_npu_core,
                "hybrid_rate": 0.0,
                "max_context": max_context,
            }
            if use_dataset:
                kwargs["dataset"] = dataset
            if use_target:
                kwargs["target_platform"] = target_platform

            desc = f"dtype={quantized_dtype} alg={alg} ({desc_suffix})"
            print(f"  Trying: {desc}...", end=" ")
            ret = llm.build(**kwargs)
            if ret == 0:
                print("✅")
                success = True
                used_dtype = quantized_dtype
                used_alg = alg
                used_target = use_target
                used_dataset = use_dataset
                break
            else:
                print(f"❌ ret={ret}")
        if success:
            break

if not success:
    print('\n❌ 所有量化类型均失败！')
    sys.exit(255)

# ── 导出 ──
dtype_label = used_dtype.upper()
out_name = (
    f"Qwen3-4B-Instruct-2507-rk3588-{dtype_label}"
    f"-opt-1-alg-{used_alg}-16k.rkllm"
)
out_path = os.path.join(os.getcwd(), out_name)

ret = llm.export_rkllm(out_path)
if ret != 0:
    print(f'\n❌ 导出失败! ret={ret}')
    sys.exit(ret)

print(f'\n✅ 导出成功: {out_name}')
print(f'   量化类型: {used_dtype}')
print(f'   算法: {used_alg}')
print(f'   上下文长度: {max_context}')

# 输出 summary
summary = {
    "model": out_name,
    "quantized_dtype": used_dtype,
    "quantized_algorithm": used_alg,
    "max_context": max_context,
    "target_platform": target_platform if used_target else "default",
    "dataset_used": used_dataset,
}
with open("export_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f'Summary saved: export_summary.json')
