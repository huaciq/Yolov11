#!/usr/bin/env python3

"""
推理脚本（YOLOv11 / Ultralytics 接口）.

用法：
- 直接运行该脚本即可，无需命令行参数
- 如需修改推理参数或输入源，请在下方 CONFIG 中直接编辑
"""

import os
import sys
from pathlib import Path

"""确保可以导入本仓库内的 ultralytics 包（无需 pip 安装）。"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

# ===== 在此处直接配置推理参数（按需修改） =====
CONFIG = {
    "model": "/root/YOLOv11/runs/train/exp/weights/best.pt",  # 替换为你的权重路径
    "source": "/root/YOLOv11/ultralytics/assets/bus.jpg",  # 可为 图片/视频/目录/摄像头编号(如 0)
    "overrides": {
        # 推理阈值与设备
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 300,
        "device": "cuda:0",  # 可选："cpu"、"cuda"、"cuda:0"
        "half": False,
        "dnn": False,
        "vid_stride": 1,
        # 保存与可视化
        "save": True,
        "save_txt": False,
        "save_conf": False,
        "save_crop": False,
        "show": False,
        "retina_masks": False,
        # 输出目录
        "project": "/root/YOLOv11/runs/predict",
        "name": "exp",
        "exist_ok": True,
    },
    # 是否以流式方式推理（True 适合长视频/摄像头，False 为普通批量推理）
    "stream": False,
}


def main() -> None:
    out_dir = CONFIG["overrides"].get("project", "runs/predict")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 初始化模型并执行推理
    model = YOLO(CONFIG["model"])
    results = model.predict(
        source=CONFIG["source"],
        stream=CONFIG["stream"],
        **{k: v for k, v in CONFIG["overrides"].items() if v is not None},
    )

    # 打印结果或简单统计
    if CONFIG["stream"]:
        count = 0
        for _ in results:
            count += 1
        print(f"流式推理完成，共处理帧/样本数：{count}")
    else:
        print(f"推理完成，样本数：{len(results) if hasattr(results, '__len__') else '未知'}")


if __name__ == "__main__":
    main()
