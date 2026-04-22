# vlm_nav.py —— 改进版：模糊匹配 + 宽松后处理
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import os

MODEL_PATH = "/home/nuaawzh/VLM/Qwen2-VL-7B-Instruct-ms"
IMAGE_DIR = "/home/nuaawzh/VLM/habitat-vlm"
MAX_STEPS = 7

print("加载 7B 模型...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
).eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("模型就绪\n")

history_actions = []
results = []

for step in range(MAX_STEPS):
    img_path = os.path.join(IMAGE_DIR, f"frame_{step:03d}.jpg")
    if not os.path.exists(img_path):
        continue
    
    print(f"--- 处理 frame_{step:03d}.jpg ---")
    image = Image.open(img_path).convert("RGB")
    
    max_size = 560
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        new_w = (new_w // 28) * 28
        new_h = (new_h // 28) * 28
        new_w, new_h = max(new_w, 28), max(new_h, 28)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    # Prompt
    history_str = "无" if not history_actions else " -> ".join(history_actions[-3:])
    prompt = """你正在一个现代公寓内，当前任务是：找到通往卧室的门。
当前是第 {step} 步，之前执行的动作：{history_str}。

观察当前画面，选择下一步动作：
- move_forward: 如果前方有门或通道
- turn_left: 如果左侧有门
- turn_right: 如果右侧有门
- stop: 如果已经找到卧室门

只回答动作名称。"""

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.3,
        )
    
    response = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()
    
    # ========== 模糊匹配解析 ==========
    response_clean = response.lower().replace(" ", "_").replace("-", "_")
    
    action_aliases = {
        "move_forward": ["move_forward", "moveforward", "forward", "foreward", "forword", "ahead", "go_forward"],
        "turn_left": ["turn_left", "turnleft", "left_turn", "left"],
        "turn_right": ["turn_right", "turnright", "right_turn", "right"],
        "stop": ["stop", "halt", "wait", "stand"],
    }
    
    action = "stop"
    for canonical, aliases in action_aliases.items():
        for alias in aliases:
            if alias in response_clean:
                action = canonical
                break
        if action != "stop":
            break
    
    # ========== 修复：先追加，再检查连续3次 ==========
    history_actions.append(action)
    
    # 只在可能撞墙/卡死时才干预，而不是为了"多样性"
    if len(history_actions) >= 3 and all(a == action for a in history_actions[-3:]):
    # 只有当前动作是 stop 或原地打转时才干预
        if action == "stop":
            action = "turn_left"  # 被困住了，转一下
    # 如果是 move_forward 或 turn_left/right，不干预，让模型自己决定
    
    print(f"  原始输出: {response}")
    print(f"  解析动作: {action}")
    results.append((step, action))
    
    
    del inputs, outputs
    torch.cuda.empty_cache()

# 输出
print("\n" + "=" * 60)
print("【VLM 导航决策结果】")
print("=" * 60)
for step, action in results:
    print(f"Step {step}: {action}")

with open(os.path.join(IMAGE_DIR, "vlm_actions.txt"), "w") as f:
    for step, action in results:
        f.write(f"{action}\n")

print(f"\n结果已保存")
print("=" * 60)