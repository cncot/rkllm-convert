"""
Qwen3-VL-4B-Instruct -> RKLLM - W8A8_G128, v1.3.0, 16K context.
Multimodal (vision-language) model for browser_use screenshot understanding.
"""
from rkllm.api import RKLLM
import os, sys, json

modelpath = os.environ.get("MODEL_PATH")
if not modelpath:
    print("MODEL_PATH required")
    sys.exit(1)

max_context = int(os.environ.get("MAX_CONTEXT", "16384"))
if max_context > 16384:
    max_context = 16384

print(f"Loading multimodal model: {modelpath}")
print(f"Max context: {max_context}")

llm = RKLLM()

# v1.3.0 supports multimodal loading
ret = llm.load_huggingface(
    model=modelpath,
    device="cpu",
    dtype="float16",
    load_weight=True
)
if ret != 0:
    print(f"Standard load failed (ret={ret}), trying multimodal path...")
    ret = llm.load_huggingface(
        model=modelpath,
        device="cpu",
        dtype="float16",
        load_weight=True,
        multimodal=True
    )
    if ret != 0:
        print(f"Multimodal load also failed! ret={ret}")
        sys.exit(ret)

print("Model loaded successfully")

print("Quantizing: w8a8 normal + G128 + target_platform (16K)...")
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
    print(f"Quantization failed! ret={ret}")
    try:
        log = llm.get_log()
        print(f"LLM log: {log}")
    except:
        pass
    sys.exit(ret)

out_name = f"Qwen3-VL-4B-Instruct-rk3588-w8a8_g128-{max_context // 1024}k.rkllm"
ret = llm.export_rkllm(out_name)
if ret != 0:
    print(f"Export failed! ret={ret}")
    sys.exit(ret)

print(f"\nOK {out_name}")
print(f"File size: {os.path.getsize(out_name) / 1e9:.2f} GB")

summary = {
    "model": "Qwen3-VL-4B-Instruct",
    "quantized_dtype": "w8a8_g128",
    "toolkit": "v1.3.0",
    "max_context": max_context,
    "target_platform": "RK3588",
    "num_npu_core": 3
}
with open("export_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
