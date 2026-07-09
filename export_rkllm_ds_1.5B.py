"""
Convert DeepSeek-R1-Distill-Qwen-1.5B to RKLLM format for RK3588 NPU.
W8A8_G128 quantization, configurable context length.

This model is based on Qwen2.5-1.5B architecture with DeepSeek R1 distillation.
It naturally produces <think>...</think> reasoning - no need for enable_thinking.

⚠️ 避坑指南（来自 v1.2.2 编译经验）:
  1. quantized_algorithm: "normal" 速度快但可能在某些架构下有 bug → 卡死时可改 "default"
  2. optimization_level: "1" 标准优化；如遇推理卡死可改 "0"（无优化，更兼容）
  3. num_npu_core: RK3588 有 3 个 NPU 核心，但某些模型在 3 核下会挂 → 可降为 2
  4. 不要使用 enable_thinking=True（DeepSeek 模型自然输出 think 标签）
  5. max_context 不要超过模型原生支持的 max_position_embeddings
"""
from rkllm.api import RKLLM
import os, sys, json

modelpath = os.environ.get("MODEL_PATH")
if not modelpath:
    print("❌ MODEL_PATH environment variable is required")
    sys.exit(1)

max_context = int(os.environ.get("MAX_CONTEXT", "16384"))
quantized_algorithm = os.environ.get("QUANT_ALGORITHM", "normal").strip()
optimization_level = int(os.environ.get("OPTIMIZATION_LEVEL", "1"))
num_npu_core = int(os.environ.get("NUM_NPU_CORE", "3"))

# Clamp max_context
if max_context < 32:
    max_context = 32
if max_context > 32768:
    max_context = 32768

print(f"📦 Model: {modelpath}")
print(f"📐 Max context: {max_context}")
print(f"🔧 Quant algo: {quantized_algorithm}")
print(f"⚡ Optimization: level {optimization_level}")
print(f"🧠 NPU cores: {num_npu_core}")
sys.stdout.flush()

# ── Step 1: Inspect model config ──
config_path = os.path.join(modelpath, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    print(f"  Architecture: {cfg.get('architectures', ['unknown'])}")
    print(f"  Model type: {cfg.get('model_type', 'unknown')}")
    print(f"  Max position embeddings: {cfg.get('max_position_embeddings', 'unknown')}")
    print(f"  Hidden size: {cfg.get('hidden_size', 'unknown')}")
    sys.stdout.flush()

# ── Step 2: Load model ──
llm = RKLLM()

print("⏳ Loading model (this may take a few minutes)...")
sys.stdout.flush()

ret = llm.load_huggingface(
    model=modelpath,
    model_lora=None,
    device="cpu",
    dtype="float16",
)
if ret != 0:
    print(f"❌ Load model failed with code {ret}!")
    sys.exit(ret)

print("✅ Model loaded successfully")
sys.stdout.flush()

# ── Step 3: Quantize & Build ──
target_platform = "rk3588"
quantized_dtype = "W8A8_G128"

print(f"⏳ Building model (this will take 2-4 hours)...")
print(f"  target_platform={target_platform}")
print(f"  quantized_dtype={quantized_dtype}")
print(f"  quantized_algorithm={quantized_algorithm}")
print(f"  optimization_level={optimization_level}")
print(f"  num_npu_core={num_npu_core}")
print(f"  max_context={max_context}")
sys.stdout.flush()

try:
    ret = llm.build(
        do_quantization=True,
        optimization_level=optimization_level,
        quantized_dtype=quantized_dtype,
        quantized_algorithm=quantized_algorithm,
        target_platform=target_platform,
        num_npu_core=num_npu_core,
        max_context=max_context,
    )
except Exception as e:
    print(f"❌ Build failed with exception: {e}")
    sys.exit(1)

if ret != 0:
    print(f"❌ Build failed with code {ret}!")
    sys.exit(ret)

print("✅ Build completed successfully!")
sys.stdout.flush()

# ── Step 4: Export ──
output = f"./DeepSeek-R1-Distill-Qwen-1.5B-rk3588-w8a8_g128-{max_context}ctx-opt{optimization_level}-{quantized_algorithm}-v122.rkllm"
if os.path.exists(output):
    os.remove(output)

print(f"⏳ Exporting to {output}...")
sys.stdout.flush()

try:
    ret = llm.export_rkllm(output)
except Exception as e:
    print(f"❌ Export failed with exception: {e}")
    sys.exit(1)

if ret != 0:
    print(f"❌ Export failed with code {ret}!")
    sys.exit(ret)

# Verify
if os.path.exists(output):
    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"✅ Export successful! File: {output}")
    print(f"   Size: {size_mb:.1f} MB")
else:
    print(f"❌ Export file not found!")
    sys.exit(1)

# Set output path for GitHub Actions
github_output = os.environ.get("GITHUB_OUTPUT", "")
if github_output:
    with open(github_output, "a") as f:
        f.write(f"model_file={output}\n")
    print(f"Set output model_file={output}")

sys.stdout.flush()
