"""
Convert Qwen3-4B-Instruct-2507 to RKLLM for RK3588 NPU.
Target: W4A16_G32 16K, with fallback to W8A8 16K if needed.
"""
from rkllm.api import RKLLM
import os, sys, json

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

# ── 策略：先试 W4A16_G32，不行就 W8A8 ──
# 算法: gdq 专门为 4-bit 优化, normal 通用
dtype_attempts = [
    ("w4a16_g32", ["gdq", "normal"]),   # 先试 4-bit 分组量化
    ("w4a16",     ["gdq", "normal"]),   # 试试不带 _g32 的 4-bit
    ("w8a8",      ["normal", "gdq"]),   # 最后回退到 W8A8
]

success = False
final_ret = -1
used_dtype = None
used_alg = None
used_target = None
used_dataset = None

for quantized_dtype, algorithms in dtype_attempts:
    if success:
        break
    for alg in algorithms:
        # 尝试：有 dataset + target
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
            sys.stdout.flush()
            ret = llm.build(**kwargs)
            if ret == 0:
                print("✅")
                sys.stdout.flush()
                success = True
                used_dtype = quantized_dtype
                used_alg = alg
                used_target = use_target
                used_dataset = use_dataset
                break
            else:
                print(f"❌ ret={ret}")
                sys.stdout.flush()
        if success:
            break

if not success:
    print('\n❌ 所有量化类型均失败！')
    print('   RKLLM Toolkit v1.2.3 不支持此模型的 W4A16_G32 或 W8A8 量化')
    sys.exit(final_ret)

# ── 输出 ──
# 标准化 dtype 名称用于文件名
dtype_label = used_dtype.upper()
out_name = (
    f"Qwen3-4B-Instruct-2507-rk3588-{dtype_label}"
    f"-opt-1-alg-{used_alg}-16k.rkllm"
)
out_path = os.path.join(os.getcwd(), out_name)

# 导出后重命名
import shutil
# 默认导出文件名是 model.rkllm
default_out = os.path.join(os.getcwd(), "model.rkllm")
if os.path.exists(default_out):
    shutil.move(default_out, out_path)
    print(f'\n✅ 导出成功: {out_name}')
    print(f'   量化类型: {used_dtype}')
    print(f'   算法: {used_alg}')
    print(f'   上下文长度: {max_context}')
else:
    # 尝试在其他位置找
    for f in os.listdir('.'):
        if f.endswith('.rkllm'):
            shutil.move(f, out_path)
            print(f'\n✅ 导出成功 (renamed): {out_name}')
            break
    else:
        print(f'\n⚠️ 模型已导出但未找到 .rkllm 文件，请手动查找')

# 输出 summary JSON
summary = {
    "model": out_name,
    "quantized_dtype": used_dtype,
    "quantized_algorithm": used_alg,
    "max_context": max_context,
    "target_platform": target_platform if used_target else "default",
    "dataset_used": used_dataset,
    "file": out_path,
}
with open("export_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f'Summary: export_summary.json')
sys.stdout.flush()
