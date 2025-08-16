#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 YOLOv11/Ultralytics 的 .pt 权重导出为 ONNX（适用于 Orange Pi 等 ARM64 设备上的 onnxruntime 推理）

用法：
- 直接运行该脚本即可，无需命令行参数
- 如需修改导出参数，在下方 CONFIG 中直接编辑

注意：
- 建议在 PC/x86 上完成导出（无需在 Orange Pi 上安装 PyTorch），再将 .onnx 文件拷贝到开发板
"""

import os
import sys
from pathlib import Path

# 确保可以导入本仓库内的 ultralytics 包（无需 pip 安装）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


# ===== 在此处直接配置导出参数（按需修改） =====
CONFIG = {
    # 输入权重（.pt）路径
    "model": "/root/YOLOv11/runs/train/exp/weights/best.pt",  # 替换为你的 .pt 权重

    # 导出配置
    "imgsz": 640,          # 导出时的输入尺寸（可为单值或 [h, w]）
    "opset": 12,           # ONNX opset，一般 12/13/17 均可
    "dynamic": False,      # 是否导出动态输入（True 兼容性更好；False 性能稳定）
    "simplify": True,      # 使用 onnx-simplifier 简化图
    "half": False,         # 半精度（大多 CPU/ORT 不建议）

    # 输出目录（若为空则使用默认 runs/export）
    "project": "/root/YOLOv11/runs/export",
    "name": "onnx",
}


def main() -> None:
    # 创建输出目录
    out_dir = CONFIG.get("project")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 加载 .pt 模型
    model = YOLO(CONFIG["model"])  # 会自动解析结构

    # 执行导出为 ONNX
    onnx_path = model.export(
        format="onnx",
        imgsz=CONFIG["imgsz"],
        opset=CONFIG["opset"],
        dynamic=CONFIG["dynamic"],
        simplify=CONFIG["simplify"],
        half=CONFIG["half"],
        project=CONFIG.get("project", None),
        name=CONFIG.get("name", None),
        exist_ok=True,
    )

    print(f"ONNX 导出完成：{onnx_path}")
    print("请将生成的 .onnx 文件拷贝到 Orange Pi（例如 /home/orangepi/models 下）")


if __name__ == "__main__":
    main()


