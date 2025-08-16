#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练脚本（YOLOv11 / Ultralytics 接口）

用法：
- 直接运行该脚本即可，无需命令行参数
- 如需修改训练参数，请在下方 CONFIG 中直接编辑
"""

import os
import sys
from pathlib import Path

"""确保可以导入本仓库内的 ultralytics 包（无需 pip 安装）。"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

# ===== 在此处直接配置训练参数（按需修改） =====
CONFIG = {
    # 模型与数据
    "model": "yolo11n.pt",  # 可填写 .pt 或 .yaml
    "overrides": {
        # 数据与基础配置
        "data": "/root/YOLOv11/ultralytics/cfg/datasets/dataset.yaml",  # 替换为你的 data.yaml
        "epochs": 100,
        "imgsz": 640,
        "batch": 128,
        "workers": 20,
        "device": "cuda:0",  # 可选："cpu"、"cuda"、"cuda:0"
        "seed": 0,
        "cache": True,

        # 输出与日志
        "project": "/root/YOLOv11/runs/train",
        "name": "exp",
        "exist_ok": True,
        "save_period": -1,
        "plots": True,

        # 优化器/调度器（常用项）
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,

        # 训练行为
        "pretrained": True,  # 若使用 .yaml 初始化并加载预训练，可设为 True
        "resume": False,      # 若继续训练（目录下需有 last.pt）
    },
}


def main() -> None:
    # 确保输出目录存在
    out_dir = CONFIG["overrides"].get("project", "runs/train")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 初始化模型（可传入 .pt 或 .yaml）
    model = YOLO(CONFIG["model"])

    # 启动训练
    results = model.train(**{k: v for k, v in CONFIG["overrides"].items() if v is not None})

    # 控制台友好输出
    print("训练完成，结果目录：", getattr(results, "save_dir", out_dir))


if __name__ == "__main__":
    main()


