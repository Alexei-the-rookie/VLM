import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import time

# ========== 7B 模型路径 ==========
MODEL_PATH = "/home/nuaawzh/VLM/Qwen2-VL-7B-Instruct-ms"
IMAGE_PATH = "corridor.jpg"

print("=" * 60)
print("加载 7B 模型（4-bit 量化）...")
print("=" * 60)

# 4-bit 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # 嵌套量化，进一步省显存
    bnb_4bit_quant_type="nf4",       # 4-bit 正常浮点，精度更好
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",  # 自动分配，如果显存不够会自动 offload 到 CPU
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
).eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("模型加载完成\n")

# ========== 加载并缩放图片 ==========
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"原图尺寸: {image.size}")

# 7B 模型必须更保守：最大边不超过 840
# 否则视觉编码器 + 7B LLM 的 KV Cache 会爆 8GB 显存
max_size = 1680
w, h = image.size
if max(w, h) > max_size:
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    # 对齐到 28 的倍数
    new_w = (new_w // 28) * 28
    new_h = (new_h // 28) * 28
    new_w, new_h = max(new_w, 28), max(new_h, 28)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"缩放后: {image.size}")
else:
    print(f"保持原尺寸: {image.size}")

# ========== 实验配置 ==========
experiments = [
    {
        "name": "实验A：导航动作决策",
        "prompt": "观察图片后，只回答一个词：直行 / 左转 / 右转 / 停止。不要解释。",
    },
    {
        "name": "实验B：空间距离估计",
        "prompt": "左侧地面的绿色瓶子，离最近的黄色门框多远？只回答：很近 / 中等距离 / 较远。",
    },
    {
        "name": "实验C：地面障碍物排序",
        "prompt": "只关注地面。从当前位置走到走廊尽头，地面上有哪些东西需要避开？按从左到右顺序列出，只写物体名称。",
    },
]

# ========== 批量推理 ==========
for exp in experiments:
    print("\n" + "=" * 60)
    print(f"【{exp['name']}】")
    print(f"Prompt: {exp['prompt']}")
    print("-" * 60)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": exp["prompt"]},
        ]
    }]
    
    # 预处理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # 推理
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,      # 7B 必须缩短，减少 KV Cache
            do_sample=True,
            temperature=0.2,         # 更低温度，更确定
            top_p=0.9,
            repetition_penalty=1.3,
        )
    elapsed = time.time() - start
    
    response = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()
    
    print(f"输出 ({elapsed:.2f}s):")
    print(response)
    
    # 清理显存
    del inputs, outputs
    torch.cuda.empty_cache()

# ========== 显存报告 ==========
print("\n" + "=" * 60)
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"显存: {allocated:.2f}GB 已分配 / {reserved:.2f}GB 预留 / {total:.2f}GB 总计")
print("=" * 60)