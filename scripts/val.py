#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
验证脚本（YOLOv11 / Ultralytics 接口）

用法：
- 直接运行该脚本即可，无需命令行参数
- 如需修改评估参数，请在下方 CONFIG 中直接编辑
"""

import os
import sys
from pathlib import Path

"""确保可以导入本仓库内的 ultralytics 包（无需 pip 安装）。"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

# ===== 在此处直接配置验证参数（按需修改） =====
CONFIG = {
    "model": "/root/YOLOv11/runs/train/exp/weights/best.pt",  # 替换为你的权重路径
    "overrides": {
        # 数据与基础配置
        "data": "/root/YOLOv11/ultralytics/cfg/datasets/dataset.yaml",  # 替换为你的 data.yaml
        "split": "val",       # 可选："val" 或 "test"
        "imgsz": 640,
        "batch": 1,
        "workers": 20,
        "device": "cuda:0",   # 可选："cpu"、"cuda"、"cuda:0"

        # 推理与评估阈值
        "conf": 0.45,
        "iou": 0.6,
        "max_det": 300,
        "half": False,
        "dnn": False,

        # 输出与日志
        "project": "/root/YOLOv11/runs/val",
        "name": "exp",
        "exist_ok": True,
        "plots": True,
        "save_json": False,
    },
}


def main() -> None:
    # 确保输出目录存在
    out_dir = CONFIG["overrides"].get("project", "runs/val")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 加载模型权重并执行验证
    model = YOLO(CONFIG["model"])
    metrics = model.val(**{k: v for k, v in CONFIG["overrides"].items() if v is not None})

    # 打印关键指标（mAP50-95 等），详细图表与日志会保存在输出目录
    print("评估完成，指标：", metrics)


if __name__ == "__main__":
    main()


