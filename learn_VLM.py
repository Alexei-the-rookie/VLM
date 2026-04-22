import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import os

MODEL_PATH = "./Qwen2-VL-2B-Instruct-ms/qwen/Qwen2-VL-2B-Instruct"

print("正在加载模型...")
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
print("模型加载完成")

# ========== 加载图片 ==========
IMAGE_PATH = "corridor.jpg"
image = Image.open(IMAGE_PATH).convert("RGB")
print(f"原图尺寸: {image.size}")

# 智能缩放：保持比例，最大边不超过672（Qwen2-VL支持的最大分辨率之一）
# 但控制在8GB显存能承受的范围内，用560比较安全
max_size = 560
w, h = image.size
if max(w, h) > max_size:
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    # 对齐到28的倍数（Qwen2-VL的patch规则）
    new_w = (new_w // 28) * 28
    new_h = (new_h // 28) * 28
    new_w, new_h = max(new_w, 28), max(new_h, 28)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"缩放后: {image.size}")
else:
    print(f"保持原尺寸: {image.size}")

# ========== 构造对话（简化prompt，减少误导） ==========
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "描述这个走廊场景。指出：1. 前方有什么；2. 左右两侧有什么；3. 地面上有什么需要注意的障碍物。"},
    ]
}]

# ========== 预处理 & 推理 ==========
print("预处理中...")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda:0")

print("推理中...")
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.3,  # 加强重复惩罚
    )
    
    response = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

print("\n" + "="*50)
print("模型输出：")
print(response)
print("="*50)

allocated = torch.cuda.memory_allocated() / 1024**3
print(f"\n显存占用: {allocated:.2f}GB")