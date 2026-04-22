import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import time

MODEL_PATH = "./Qwen2-VL-2B-Instruct-ms/qwen/Qwen2-VL-2B-Instruct"
IMAGE_PATH = "corridor.jpg"

# ========== 加载模型 ==========
print("=" * 60)
print("加载模型...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    quantization_config=quantization_config,
).eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("模型加载完成\n")

# ========== 加载并缩放图片 ==========
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"原图尺寸: {image.size}")

max_size = 1568  # 增大此处限制以提升输入分辨率，可根据显存情况调整至1568或更大
w, h = image.size
if max(w, h) > max_size:
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w = (new_w // 28) * 28
    new_h = (new_h // 28) * 28
    new_w, new_h = max(new_w, 28), max(new_h, 28)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"缩放后: {image.size}\n")
else:
    print(f"保持原尺寸: {image.size}\n")

# ========== 定义三个实验 ==========
experiments = [
    {
        "name": "实验A：导航动作决策",
        "prompt": "假设你是一个机器人，当前面向走廊前方。请给出下一步行动建议：直行、左转、右转，还是停止？简要说明原因。",
    },
    {
        "name": "实验B：空间距离估计",
        "prompt": "图中左侧地面的绿色瓶子距离黄色门框大约多远？用'很近'、'中等距离'、'较远'描述。",
    },
    {
        "name": "实验C：障碍物检测与排序",
        "prompt": "如果要从当前位置直线走到走廊尽头，地面上有哪些物体需要避开？按从左到右的顺序列出。",
    },
]

# ========== 批量推理 ==========
results = []

for exp in experiments:
    print("=" * 60)
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
    ).to("cuda:0")
    
    # 推理计时
    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.3,
        )
    elapsed = time.time() - start
    
    # 解码
    response = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()
    
    print(f"输出 ({elapsed:.2f}s):")
    print(response)
    print()
    
    results.append({
        "name": exp["name"],
        "prompt": exp["prompt"],
        "response": response,
        "time": elapsed,
    })

# ========== 汇总报告 ==========
print("=" * 60)
print("【实验汇总】")
print("=" * 60)

for r in results:
    print(f"\n{r['name']} ({r['time']:.2f}s)")
    print(f"  Prompt: {r['prompt'][:50]}...")
    print(f"  输出: {r['response'][:100]}...")

# 显存信息
allocated = torch.cuda.memory_allocated() / 1024**3
print(f"\n最终显存占用: {allocated:.2f}GB")
print("=" * 60)