# loop-habitat.py —— 修复路径，换 apartment_1 场景
import habitat_sim
import numpy as np
from PIL import Image

# ========== 配置 ==========
# 修正路径 + 换 apartment_1 场景
SCENE_PATH = "/home/nuaawzh/VLM/data/versioned_data/habitat_test_scenes/apartment_1.glb"

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = SCENE_PATH
sim_cfg.enable_physics = False

# 传感器配置
camera_sensor_spec = habitat_sim.CameraSensorSpec()
camera_sensor_spec.uuid = "rgb"
camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
camera_sensor_spec.resolution = [480, 640]
camera_sensor_spec.position = [0.0, 1.5, 0.0]
camera_sensor_spec.orientation = [0.0, 0.0, 0.0]

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [camera_sensor_spec]

# 创建仿真器
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
agent = sim.initialize_agent(0)

# ========== 辅助函数 ==========
def save_observation(sim, step):
    obs = sim.get_sensor_observations()
    rgb = obs["rgb"][:, :, :3]
    img = Image.fromarray(rgb)
    img.save(f"frame_{step:03d}.jpg")
    print(f"✓ 保存 frame_{step:03d}.jpg ({img.size[0]}x{img.size[1]})")
    return img

def get_agent_state(agent):
    state = agent.get_state()
    pos = state.position
    print(f"  位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    return state

def do_action(sim, agent_id, action_name):
    sim.step({agent_id: action_name})

# ========== 运行循环 ==========
print("=" * 60)
print("Habitat 仿真启动")
print(f"场景: {SCENE_PATH}")
print("=" * 60)

# 初始帧
print("\n--- Step 0: 初始位置 ---")
save_observation(sim, 0)
get_agent_state(agent)

# 动作序列
actions = [
    ("move_forward", "前进"),
    ("move_forward", "前进"),
    ("turn_left", "左转"),
    ("move_forward", "前进"),
    ("turn_right", "右转"),
    ("move_forward", "前进"),
]

for i, (action_name, desc) in enumerate(actions, 1):
    print(f"\n--- Step {i}: {desc} ---")
    do_action(sim, 0, action_name)
    save_observation(sim, i)
    get_agent_state(agent)

sim.close()
print("\n" + "=" * 60)
print("仿真结束")
print("=" * 60)