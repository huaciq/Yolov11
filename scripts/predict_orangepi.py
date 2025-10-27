#!/usr/bin/env python3

"""
在 Orange Pi（ARM64）上使用 ONNXRuntime 进行 YOLOv11 推理的示例脚本。.

建议流程： 1) 在 PC 上运行 scripts/export_onnx.py 导出 .onnx 模型 2) 将 .onnx 文件与本脚本拷贝到 Orange Pi 3) 在 Orange Pi 上安装
onnxruntime、opencv-python 等依赖 4) 运行本脚本进行图片/视频/摄像头推理

本脚本仅依赖 onnxruntime + opencv，不依赖 PyTorch 与 ultralytics。
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ===== 在此处直接配置推理参数（按需修改） =====
CONFIG = {
    # 模型与类别
    "onnx_model": "/root/YOLOv11/runs/train/exp/weights/best.onnx",  # 请替换为 Orange Pi 上的实际路径
    "class_names": None,  # 可传入类别名称列表；为 None 时以数字索引显示
    # 输入与预处理
    "imgsz": (640, 640),  # (w, h)
    "conf_thres": 0.25,
    "iou_thres": 0.7,
    "use_letterbox": True,  # 是否信箱缩放，保持纵横比
    # 设备/性能
    "providers": [
        "CPUExecutionProvider"
        # 若 Orange Pi 支持 ARM NN / OpenVINO / CoreML，可按需增添对应 EP
    ],
    # I/O 源
    "source": "/home/orangepi/samples/bus.jpg",  # 可为 图片/视频/目录/摄像头编号(如 0)
    "save_vis": True,
    "save_dir": "/home/orangepi/output",
}


def letterbox(
    im: np.ndarray, new_shape: tuple[int, int], color=(114, 114, 114)
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """信箱缩放：保持纵横比缩放到目标尺寸，并在空白处填充灰色边。返回 (图像, 缩放比例, (pad_w, pad_h))."""
    h, w = im.shape[:2]
    new_w, new_h = new_shape
    r = min(new_w / w, new_h / h)
    nw, nh = round(w * r), round(h * r)
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    dw, dh = (new_w - nw) // 2, (new_h - nh) // 2
    canvas[dh : dh + nh, dw : dw + nw] = im_resized
    return canvas, r, (dw, dh)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """简易 NMS（适合少量框）。返回保留的索引列表。."""
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return keep


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """计算单个框与多框的 IoU。."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def preprocess_bgr(image_bgr: np.ndarray, imgsz: tuple[int, int], use_letterbox: bool) -> tuple[np.ndarray, dict]:
    """将 BGR 图像预处理为 NCHW float32/0-1，并返回元数据用于反变换坐标。."""
    if use_letterbox:
        lb, r, (dw, dh) = letterbox(image_bgr, imgsz)
        meta = {"ratio": r, "pad": (dw, dh), "shape": image_bgr.shape[:2]}
        im = lb
    else:
        h0, w0 = image_bgr.shape[:2]
        im = cv2.resize(image_bgr, imgsz, interpolation=cv2.INTER_LINEAR)
        meta = {"ratio_hw": (imgsz[1] / h0, imgsz[0] / w0), "shape": (h0, w0)}

    im = im.astype(np.float32) / 255.0
    im = im.transpose(2, 0, 1)[None, ...]  # NCHW
    return im, meta


def postprocess(
    outputs: list[np.ndarray], meta: dict, conf_thres: float, iou_thres: float, num_classes: int
) -> list[np.ndarray]:
    """
    适配常见 YOLO 导出ONNX的输出格式：

    - 假设输出为 (N, num_dets, 4+num_classes)，坐标为 [cx, cy, w, h] 或 [x1, y1, x2, y2]
    - 这里先尝试识别格式；若与你的导出不同，请按需调整。
    返回：每个检测 [x1, y1, x2, y2, score, cls].
    """
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]
    # 猜测是否为中心点格式（cx,cy,w,h）
    is_cxcywh = np.all(pred[:5, :4] >= 0) and np.all(pred[:5, :4] <= 1) is False  # 仅做粗略判断

    boxes = pred[:, :4].copy()
    scores = pred[:, 4:]  # 类别置信度
    cls_ids = scores.argmax(axis=1)
    confs = scores.max(axis=1)

    mask = confs >= conf_thres
    boxes, confs, cls_ids = boxes[mask], confs[mask], cls_ids[mask]
    if boxes.size == 0:
        return []

    # 将 [cx,cy,w,h] 转为 [x1,y1,x2,y2]（若需要）
    if is_cxcywh:
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

    # 从模型输入空间反变换到原图坐标
    if "ratio" in meta:
        r = meta["ratio"]
        dw, dh = meta["pad"]
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= r
    elif "ratio_hw" in meta:
        rh, rw = meta["ratio_hw"]
        boxes[:, [0, 2]] /= rw
        boxes[:, [1, 3]] /= rh

    # NMS
    keep = nms(boxes, confs, iou_thres)
    boxes, confs, cls_ids = boxes[keep], confs[keep], cls_ids[keep]

    dets = np.concatenate([boxes, confs[:, None], cls_ids[:, None].astype(np.float32)], axis=1)
    return dets.tolist()


def visualize(image_bgr: np.ndarray, detections: list[list[float]], class_names: list[str] | None = None) -> np.ndarray:
    """在图像上绘制检测结果。detections: [x1,y1,x2,y2,score,cls]."""
    im = image_bgr.copy()
    for x1, y1, x2, y2, s, c in detections:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cls_id = int(c)
        label = f"{class_names[cls_id] if class_names else cls_id}:{s:.2f}"
        cv2.rectangle(im, p1, p2, (0, 255, 0), 2)
        cv2.putText(im, label, (p1[0], max(p1[1] - 2, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return im


def infer_on_image(session: ort.InferenceSession, image_path: str) -> np.ndarray:
    """对单张图片进行推理并返回可视化结果。."""
    img = cv2.imread(image_path)
    assert img is not None, f"无法读取图像: {image_path}"
    inp, meta = preprocess_bgr(img, CONFIG["imgsz"], CONFIG["use_letterbox"])

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, {input_name: inp})

    # 猜测类别数
    pred0 = outputs[0][0]
    num_classes = pred0.shape[-1] - 4 if pred0.ndim == 2 else pred0.shape[1] - 4
    dets = postprocess(outputs, meta, CONFIG["conf_thres"], CONFIG["iou_thres"], num_classes)
    vis = visualize(img, dets, CONFIG["class_names"])
    return vis


def main() -> None:
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    sess_options = ort.SessionOptions()
    # 可选：减少线程以降低发热
    # sess_options.intra_op_num_threads = 2
    # sess_options.inter_op_num_threads = 2

    session = ort.InferenceSession(CONFIG["onnx_model"], sess_options, providers=CONFIG["providers"])

    src = CONFIG["source"]
    if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
        cap = cv2.VideoCapture(int(src))
        assert cap.isOpened(), f"无法打开摄像头: {src}"
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            inp, meta = preprocess_bgr(frame, CONFIG["imgsz"], CONFIG["use_letterbox"])
            input_name = session.get_inputs()[0].name
            outputs = session.run([o.name for o in session.get_outputs()], {input_name: inp})
            pred0 = outputs[0][0]
            num_classes = pred0.shape[-1] - 4 if pred0.ndim == 2 else pred0.shape[1] - 4
            dets = postprocess(outputs, meta, CONFIG["conf_thres"], CONFIG["iou_thres"], num_classes)
            vis = visualize(frame, dets, CONFIG["class_names"])
            cv2.imshow("YOLOv11-ONNX", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        p = Path(src)
        if p.is_dir():
            for img_path in sorted(p.glob("*.*")):
                vis = infer_on_image(session, str(img_path))
                save_path = str(Path(CONFIG["save_dir"]) / f"{img_path.stem}_pred.jpg")
                cv2.imwrite(save_path, vis)
                print(f"保存：{save_path}")
        else:
            # 单张图片或视频文件
            ext = p.suffix.lower()
            if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
                vis = infer_on_image(session, str(p))
                save_path = str(Path(CONFIG["save_dir"]) / f"{p.stem}_pred.jpg")
                cv2.imwrite(save_path, vis)
                print(f"保存：{save_path}")
            else:
                cap = cv2.VideoCapture(str(p))
                assert cap.isOpened(), f"无法打开视频: {p}"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_path = str(Path(CONFIG["save_dir"]) / f"{p.stem}_pred.mp4")
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    inp, meta = preprocess_bgr(frame, CONFIG["imgsz"], CONFIG["use_letterbox"])
                    input_name = session.get_inputs()[0].name
                    outputs = session.run([o.name for o in session.get_outputs()], {input_name: inp})
                    pred0 = outputs[0][0]
                    num_classes = pred0.shape[-1] - 4 if pred0.ndim == 2 else pred0.shape[1] - 4
                    dets = postprocess(outputs, meta, CONFIG["conf_thres"], CONFIG["iou_thres"], num_classes)
                    vis = visualize(frame, dets, CONFIG["class_names"])
                    writer.write(vis)
                writer.release()
                cap.release()
                print(f"保存：{out_path}")


if __name__ == "__main__":
    main()
