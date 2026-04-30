#!/usr/bin/env python
"""
loop_navigation.py —— Habitat + VLM 循环导航框架 (v2)

工作流程：
  1. 初始化 Habitat 仿真器 + VLM 模型（一次性加载）
  2. 循环：
      a. 从 Habitat 获取当前帧并保存（JPEG quality=95）
      b. VLM 看图推理（分辨率 784×max），输出导航动作
      c. Habitat 执行动作（转向×1，前进×5）
      d. 检查终止条件（达到 MAX_STEPS 或 stop）
  3. 输出导航轨迹报告
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image

# ===== Habitat 相关 =====
import habitat_sim

# ===== VLM 相关 =====
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ========== 配置 ==========

# 2B 模型路径（8GB GPU 显存刚好够用）
MODEL_PATH = "/home/nuaawzh/VLM/Qwen2-VL-2B-Instruct-ms/qwen/Qwen2-VL-2B-Instruct"
# 可用显存上限（预留1.5GiB给渲染和系统）
GPU_MAX_MEMORY = 6_000_000_000  # ~6GiB

# ===== 场景配置 =====
# 在下方修改 SCENE_NAME 即可切换场景
SCENES = {
    "apartment": {
        "path": "/home/nuaawzh/VLM/data/versioned_data/habitat_test_scenes/apartment_1.glb",
        "task": "找到通往卧室的门",
        "max_steps": 30,
    },
    "vangogh": {
        "path": "/home/nuaawzh/VLM/data/versioned_data/habitat_test_scenes/van-gogh-room.glb",
        "task": "探索整个房间",
        "max_steps": 30,
    },
    "castle": {
        "path": "/home/nuaawzh/VLM/data/versioned_data/habitat_test_scenes/skokloster-castle.glb",
        "task": "找到城堡的大门",
        "max_steps": 50,
    },
}

# 选择场景（可选: "apartment", "vangogh", "castle"）
SCENE_NAME = "castle"
SCENE_PATH = SCENES[SCENE_NAME]["path"]
TASK_DESCRIPTION = SCENES[SCENE_NAME]["task"]

# 输出目录（不同场景的输出隔离）
OUTPUT_DIR = f"output_{SCENE_NAME}"

# 循环控制
MAX_STEPS = SCENES[SCENE_NAME]["max_steps"]  # 最大VLM决策步数（场景自适应）
STEPS_PER_ACTION = 5         # 每个VLM动作重复执行的habitat步数

# 历史动作入 prompt 的长度
HISTORY_LEN = 6

# ========== 1. 初始化 Habitat 仿真器 ==========

def init_habitat(scene_path):
    """创建 Habitat 仿真器并返回 (sim, agent_id)"""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    camera_sensor_spec = habitat_sim.CameraSensorSpec()
    camera_sensor_spec.uuid = "rgb"
    camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    camera_sensor_spec.resolution = [480, 640]
    camera_sensor_spec.position = [0.0, 1.5, 0.0]
    camera_sensor_spec.orientation = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [camera_sensor_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    agent_id = sim.initialize_agent(0)

    # 获取初始状态
    state = sim.get_agent(0).get_state()
    print(f"  Habitat 初始化完成")
    print(f"  初始位置: ({state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f})")
    return sim, state

def get_image(sim, step):
    """从 Habitat 获取当前帧 RGB 图像"""
    obs = sim.get_sensor_observations()
    rgb = obs["rgb"][:, :, :3]
    img = Image.fromarray(rgb)
    return img

def save_image(img, step, output_dir=OUTPUT_DIR):
    """保存图像到文件（quality=95 减少JPEG失真）"""
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"frame_{step:03d}.jpg")
    img.save(fname, quality=95)
    return fname

def do_action(sim, agent_id, action_name):
    """在 Habitat 中执行动作"""
    sim.step({agent_id: action_name})
    state = sim.get_agent(0).get_state()
    return state

# ========== 2. 初始化 VLM 模型 ==========

def init_vlm(model_path, gpu_max_memory=GPU_MAX_MEMORY):
    """加载 2B VLM 模型（固定 GPU 显存上限，避免与渲染冲突）"""
    print(f"\n加载 VLM 模型 ({model_path}) ...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 限制 GPU 显存，避免与渲染冲突
    max_memory = {0: gpu_max_memory}
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        max_memory=max_memory,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        offload_folder="/tmp/vlm_offload",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path)
    print(f"  ✓ 2B 模型加载成功")
    return model, processor

def preprocess_image(image, max_size=784):
    """缩放图像到合适尺寸（28的倍数），保留更多室内场景细节"""
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        new_w = (new_w // 28) * 28
        new_h = (new_h // 28) * 28
        new_w, new_h = max(new_w, 28), max(new_h, 28)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image

# ========== 3. VLM 推理 -> 动作解析 ==========

# 动作别名映射表
ACTION_ALIASES = {
    "move_forward": ["move_forward", "moveforward", "forward", "foreward", "forword", "ahead", "go_forward"],
    "turn_left":    ["turn_left", "turnleft", "left_turn", "left"],
    "turn_right":   ["turn_right", "turnright", "right_turn", "right"],
    "stop":         ["stop", "halt", "wait", "stand", "done", "finished"],
}

def parse_action(response_text):
    """
    模糊匹配 VLM 输出中的动作关键词。
    如果 VLM 输出了乱码/废话/分析文字，默认返回 move_forward（不死锁）。
    """
    text = response_text.lower().replace(" ", "_").replace("-", "_")

    # 优先匹配 stop（精确匹配，防止 "door" 被匹配）
    for alias in ["stop", "halt", "done", "finished", "stand"]:
        if alias in text:
            # 检查是不是独立词（不是 "stopped" 之类的衍生词）
            if text.strip(".,!?;:'\"()[]{}") == alias or f"_{alias}" in text or text.startswith(alias):
                return "stop"

    # 匹配导航动作
    for canonical, aliases in ACTION_ALIASES.items():
        if canonical == "stop":
            continue
        for alias in aliases:
            if alias in text:
                return canonical

    # 默认不停止，继续前进
    return "move_forward"

def build_prompt(step, history_actions, task_description, scene_name=SCENE_NAME):
    """构建 VLM 的 prompt（引导 2B 模型有效探索，移除 stop 选项由 MAX_STEPS 控制终止）"""
    if history_actions:
        recent = history_actions[-HISTORY_LEN:]
        history_str = " -> ".join(recent)
    else:
        history_str = "无"

    scene_labels = {
        "apartment": "公寓",
        "vangogh": "梵高房间",
        "castle": "城堡",
    }
    scene_label = scene_labels.get(scene_name, scene_name)

    prompt = f"""你是一个在{scene_label}里{task_description}的机器人。
第 {step} 步，之前动作：{history_str}。

观察画面，分析周围环境结构，然后选择下一步动作：

- 如果前方看起来是走廊或开放空间 → move_forward
- 如果前方是墙壁，左侧有空间 → turn_left
- 如果前方是墙壁，右侧有空间 → turn_right

回答格式：只输出一个动作名称。"""
    return prompt

def vlm_infer(model, processor, image, prompt):
    """执行 VLM 推理，返回动作字符串 + 原始输出"""
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
            max_new_tokens=64,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.3,
        )

    response = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    action = parse_action(response)

    # 清理显存
    del inputs, outputs
    torch.cuda.empty_cache()

    return action, response

# ========== 4. 主循环 ==========

def main():
    print("=" * 70)
    print("Habitat + VLM 循环导航框架 v2")
    print("=" * 70)
    print(f"场景: {SCENE_NAME}  |  任务: {TASK_DESCRIPTION}")

    # --- 初始化 Habitat ---
    print("\n[1/4] 初始化 Habitat 仿真器 ...")
    sim, _ = init_habitat(SCENE_PATH)
    AGENT_ID = 0

    # --- 初始化 VLM ---
    print("\n[2/4] 加载 VLM 模型 ...")
    model, processor = init_vlm(MODEL_PATH)
    print("  使用模型: Qwen2-VL-2B (4-bit)")

    # --- VLM 预热 ---
    print("\n[3/4] VLM 预热推理（空转一次）...")
    dummy_img = Image.new("RGB", (224, 224), color="gray")
    dummy_prompt = "回答一个动作名称：move_forward"
    warmup_action, warmup_raw = vlm_infer(model, processor, dummy_img, dummy_prompt)
    print(f"  预热完成，输出: '{warmup_raw}' -> 动作: {warmup_action}")
    del dummy_img

    # --- 主循环 ---
    print("\n[4/4] 开始导航循环")
    print("=" * 70)

    history_actions = []   # 所有历史动作
    trajectory = []        # 轨迹记录 (step, action, position_x, position_z)

    task_description = TASK_DESCRIPTION
    scene_name = SCENE_NAME

    consecutive_forward = 0  # 连续前进计数器（撞墙检测）
    stop_triggered = False   # VLM 是否输出 stop

    for step in range(MAX_STEPS + 1):
        print(f"\n--- Step {step} ---")

        # 4a. 获取当前帧并保存
        img = get_image(sim, step)
        fname = save_image(img, step)
        state = sim.get_agent(0).get_state()
        pos = state.position
        print(f"  位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | 图片: {fname} [保存品质:95]")

        # 4b. VLM 推理
        if step < MAX_STEPS:
            # 缩放图片（保留更多细节）
            img_resized = preprocess_image(img, max_size=784)
            img_w, img_h = img_resized.size
            print(f"  VLM 输入尺寸: {img_w}x{img_h}")

            # 构建 prompt
            prompt = build_prompt(step, history_actions, task_description, scene_name)

            # VLM 推理
            t0 = time.time()
            action, raw_output = vlm_infer(model, processor, img_resized, prompt)
            t1 = time.time()
            print(f"  VLM推理: {t1-t0:.2f}s | 原始输出: \"{raw_output[:80]}\"")
            print(f"  解析动作: {action}")

            # 记录动作
            history_actions.append(action)
            trajectory.append((step, action, float(pos[0]), float(pos[2])))

            # 4c. stop 检查
            if action == "stop":
                print(f"  → VLM 输出 stop，终止导航")
                stop_triggered = True
                break

            # 4d. 连续前进撞墙检测
            if action == "move_forward":
                consecutive_forward += 1
            else:
                consecutive_forward = 0

            if consecutive_forward >= 3:
                # 可能撞墙了，强制转向
                forced_action = "turn_left"
                print(f"  → 连续 {consecutive_forward} 次前进，强制转向探索")
                action = forced_action
                history_actions[-1] = forced_action
                consecutive_forward = 0

            # 4e. Habitat 执行动作
            repeat = 1 if action in ("turn_left", "turn_right") else STEPS_PER_ACTION
            new_state = state
            for _ in range(repeat):
                new_state = do_action(sim, AGENT_ID, action)
            new_pos = new_state.position
            dx = new_pos[0] - state.position[0]
            dz = new_pos[2] - state.position[2]
            dist = (dx**2 + dz**2)**0.5
            print(f"  执行 {action} ×{repeat} → ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f}) 移动: {dist:.2f}")
        else:
            print(f"  达到最大步数 {MAX_STEPS}，停止导航")
            trajectory.append((step, "stop", float(pos[0]), float(pos[2])))
            break

    # --- 收尾 ---
    sim.close()
    print("\n" + "=" * 70)
    print("导航结束")
    print("=" * 70)

    print(f"\n【导航轨迹 ({len(trajectory)} 步)】")
    print(f"{'Step':<6} {'动作':<16} {'位置X':<10} {'位置Z':<10}")
    print("-" * 42)
    for s, act, px, pz in trajectory:
        print(f"{s:<6} {act:<16} {px:<10.2f} {pz:<10.2f}")

    report_path = os.path.join(OUTPUT_DIR, "navigation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Habitat + VLM 导航报告 (v2)\n")
        f.write(f"场景: {SCENE_NAME} ({SCENE_PATH})\n")
        f.write(f"模型: Qwen2-VL-2B (4-bit)\n")
        f.write(f"任务: {task_description}\n")
        f.write(f"最大步数: {MAX_STEPS}\n")
        f.write(f"实际步数: {len(trajectory)}\n")
        f.write(f"VLM 提前stop: {'是' if stop_triggered else '否'}\n")
        f.write(f"-" * 50 + "\n")
        f.write(f"{'Step':<6} {'动作':<16} {'位置X':<10} {'位置Z':<10}\n")
        f.write("-" * 42 + "\n")
        for s, act, px, pz in trajectory:
            f.write(f"{s:<6} {act:<16} {px:<10.2f} {pz:<10.2f}\n")
    print(f"\n报告已保存: {report_path}")
    print("=" * 70)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"显存占用: {allocated:.2f} GB")

if __name__ == "__main__":
    main()
