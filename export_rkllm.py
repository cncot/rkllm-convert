from rkllm.api import RKLLM
import os

modelpath = os.environ.get('MODEL_PATH', '/path/to/model')
device = os.environ.get('RKLLM_DEVICE', 'cuda')
dtype = os.environ.get('RKLLM_DTYPE', 'float16')

llm = RKLLM()

ret = llm.load_huggingface(model=modelpath, model_lora=None,
                           device=device, dtype=dtype,
                           custom_config=None, load_weight=True)
if ret != 0:
    print('Load model failed!')
    exit(ret)

dataset = "./data_quant.json"
target_platform = "RK3588"
quantized_dtype = "W8A8"
quantized_algorithm = "normal"
num_npu_core = 3
max_context = int(os.environ.get('MAX_CONTEXT', '4096'))

ret = llm.build(do_quantization=True, optimization_level=1,
                quantized_dtype=quantized_dtype,
                quantized_algorithm=quantized_algorithm,
                target_platform=target_platform,
                num_npu_core=num_npu_core,
                dataset=dataset,
                hybrid_rate=0, max_context=max_context)
if ret != 0:
    print('Build model failed!')
    exit(ret)

out_name = f"{os.path.basename(modelpath)}_{quantized_dtype}_{target_platform}.rkllm"
ret = llm.export_rkllm(f"./{out_name}")
if ret != 0:
    print('Export model failed!')
    exit(ret)

print(f'Export success: {out_name}')
