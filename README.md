# VLM Navigation Prototype

Habitat-Sim + Qwen2-VL 闭环视觉导航原型系统。

## 框架流程

```
Habitat-Sim 仿真器          Qwen2-VL 大模型
┌────────────────────┐    ┌──────────────────┐
│ 第一视角 RGB 图像    │ →→ │ 看图 → 输出动作    │
│ 执行动作（前/左/右）  │ ←← │ (move_forward /  │
│ 更新 Agent 位置     │    │  turn_left /     │
└────────────────────┘    │  turn_right)     │
                          └──────────────────┘
```

单进程内实时循环，无需文件中转。

## 目录结构

```
VLM/
├── habitat-vlm/
│   ├── loop_navigation.py          # 核心：闭环导航框架
│   ├── loop-habitat.py             # 遗留：纯 Habitat 生成图片
│   ├── output_apartment/           # 公寓场景测试结果
│   ├── output_vangogh/             # 梵高房间测试结果
│   └── output_castle/              # 城堡场景测试结果
├── vlm_nav.py                      # 遗留：纯 VLM 推理
├── README.md                       # 本文件
├── README(26-04-22).md             # 阶段一报告
├── README(26-04-30).md             # 阶段二报告
├── Qwen2-VL-2B-Instruct-ms/        # 2B 模型（不上传 Git）
├── Qwen2-VL-7B-Instruct-ms/        # 7B 模型（不上传 Git）
└── data/                           # Habitat 场景数据（不上传 Git）
```

## 快速开始

### 环境配置

```bash
conda create -n habitat python=3.9 cmake=3.14.0 -y
conda activate habitat

# Habitat-Sim
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y

# VLM 推理依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==5.5.4 accelerate qwen-vl-utils Pillow bitsandbytes

# 下载测试场景
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

### 运行闭环导航

```bash
conda activate habitat
cd habitat-vlm
python loop_navigation.py
```

在脚本顶部修改 `SCENE_NAME` 即可切换场景（`apartment` / `vangogh` / `castle`）。

## 支持的场景

| 场景 | 名称 | 步数 | 任务 |
|------|------|------|------|
| 现代公寓 | `apartment` | 30 | 找到通往卧室的门 |
| 梵高房间 | `vangogh` | 30 | 探索整个房间 |
| 斯德哥尔摩城堡 | `castle` | 50 | 找到城堡的大门 |

## 关键特性

- **单进程实时闭环**：Habitat 渲染 → VLM 推理 → 执行动作，无需文件通信
- **2B 模型 4-bit 量化**：显存占用约 1.5 GB，RTX 4060 Laptop (8GB) 流畅运行
- **输出隔离**：每场景自动创建 `output_{场景名}/` 目录，保存图片 + 轨迹报告
- **撞墙检测**：连续 3 次前进不可移动时强制转向，避免死锁
- **场景配置化**：改一行 `SCENE_NAME` 即可切换场景

## 详细文档

- [阶段一报告](README(26-04-22).md) — 两段式框架搭建、模型对比
- [阶段二报告](README(26-04-30).md) — 单进程闭环改造、三次全场景跑测

## 参考

- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
