"""
Convert Qwen3-1.7B-Instruct to RKLLM format for RK3588 NPU.
W8A8 quantization, 16K context, enable_thinking=True.

enable_thinking=True: model outputs <think>...</think> reasoning before answer.
This is configured by modifying the model's generation_config.json
before loading with RKLLM Toolkit, ensuring the thinking capability
is baked into the converted model.
"""
from rkllm.api import RKLLM
import os, sys, json

modelpath = os.environ.get("MODEL_PATH")
if not modelpath:
    print("MODEL_PATH environment variable is required")
    sys.exit(1)

max_context = int(os.environ.get("MAX_CONTEXT", "16384"))
if max_context > 16384:
    print(f"Clamping max_context from {max_context} to 16384 (16K target)")
    max_context = 16384
if max_context < 32:
    max_context = 32

enable_thinking = os.environ.get("ENABLE_THINKING", "true").lower() == "true"

print(f"Loading model from: {modelpath}")
print(f"Max context: {max_context}")
print(f"enable_thinking: {enable_thinking}")
sys.stdout.flush()

# ── Step 1: Enable thinking in model's generation config ──
gen_config_path = os.path.join(modelpath, "generation_config.json")
if os.path.exists(gen_config_path):
    with open(gen_config_path, "r") as f:
        gen_config = json.load(f)

    modified = False
    # Qwen3 uses enable_thinking in generation_config
    if gen_config.get("enable_thinking") != enable_thinking:
        gen_config["enable_thinking"] = enable_thinking
        modified = True
        print(f"  Set enable_thinking={enable_thinking} in generation_config.json")

    # Ensure top_k is set (RKLLM Toolkit needs it)
    if "top_k" not in gen_config:
        gen_config["top_k"] = 50
        modified = True

    if modified:
        with open(gen_config_path, "w") as f:
            json.dump(gen_config, f, indent=2)
        print(f"  Updated generation_config.json")
    else:
        print(f"  generation_config.json already has enable_thinking={enable_thinking}")
else:
    print(f"  WARNING: generation_config.json not found at {gen_config_path}")
    print(f"  Thinking may not be properly configured")

sys.stdout.flush()

# ── Step 2: Load model with RKLLM Toolkit ──
llm = RKLLM()

custom_config = {
    "enable_thinking": enable_thinking,
}

ret = llm.load_huggingface(
    model=modelpath,
    model_lora=None,
    device="cpu",
    dtype="float16",
    custom_config=custom_config,
    load_weight=True,
)
if ret != 0:
    print("Load model failed!")
    sys.exit(ret)

print("Model loaded successfully")
sys.stdout.flush()

# ── Step 3: Quantize ──
dataset = "./data_quant.json"
target_platform = "RK3588"
quantized_dtype = "W8A8"
quantized_algorithm = "normal"
num_npu_core = 3
optimization_level = 1

print(f"Starting quantization: max_context={max_context}, dtype={quantized_dtype}...")
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
        max_context=max_context,
    )
except Exception as e:
    print(f"Build raised exception: {e}")
    try:
        log = llm.get_log()
        print(f"LLM log: {log}")
    except Exception:
        pass
    sys.exit(1)

if ret != 0:
    print(f"Build model failed! ret={ret}")
    try:
        output = llm.get_log()
        print(f"LLM log:\n{output}")
    except Exception as e:
        print(f"Could not get log: {e}")
    sys.exit(ret)

# ── Step 4: Export ──
out_name = (
    f"Qwen3-1.7B-thinking-rk3588-{quantized_dtype.lower()}_g128"
    f"-opt-{optimization_level}-hybrid-ratio-0.0-{max_context // 1024}k.rkllm"
)
ret = llm.export_rkllm(f"./{out_name}")
if ret != 0:
    print("Export model failed!")
    sys.exit(ret)

print(f"\n✅ Export success: {out_name}")
ls_out = os.popen("ls -lh *.rkllm").read()
print(ls_out)

# ── Step 5: Write summary ──
summary = {
    "model": out_name,
    "base_model": "Qwen/Qwen3-1.7B",
    "dtype": quantized_dtype.lower(),
    "toolkit": "v1.2.2",
    "max_context": max_context,
    "enable_thinking": enable_thinking,
    "num_npu_core": num_npu_core,
    "target_platform": target_platform,
}
with open("export_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary: {json.dumps(summary, indent=2)}")
