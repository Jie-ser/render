#!/usr/bin/env python3
"""
SAM3 批量遮罩提取脚本

在 sam3 conda 环境中运行。由 run_pipeline.py 自动调用。
输入: --tasks <tasks.json>，每条任务包含 video_path、sam3_prompt、output_path。
输出: 每个视频对应一个 RGB 遮罩视频（机器人像素保留原色，背景全黑）。
"""

import argparse
import json
import os
import subprocess
import sys
import traceback

# ------------------------------------------------------------------
# 依赖检查
# ------------------------------------------------------------------

def ensure_deps():
    missing = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        # The pipeline only needs OpenCV's video/image APIs, not GUI support.
        missing.append("opencv-python-headless")
    try:
        import imageio  # noqa: F401
    except ImportError:
        missing.append("imageio")
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        missing.append("imageio-ffmpeg")
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        # SAM3 currently imports pkg_resources; setuptools 81+ no longer
        # provides it, so keep the compatible setuptools release available.
        missing.append("setuptools<81")
    if missing:
        print(f"[extract_masks] 安装缺失依赖: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + missing
        )


ensure_deps()

import cv2
import imageio
import numpy as np
import torch  # noqa: F401
from sam3.model_builder import build_sam3_predictor


# ------------------------------------------------------------------
# 核心逻辑
# ------------------------------------------------------------------

def extract_one(predictor, video_path, prompt, output_path):
    """对单个视频做 SAM3 分割，输出遮罩视频。

    遮罩视频格式: RGB，机器人区域保留原始像素颜色，非机器人区域为纯黑。
    返回 (总帧数, 有遮罩的帧数)。
    """

    # ---------- SAM3 推理 ----------
    resp = predictor.handle_request(
        dict(type="start_session", resource_path=video_path)
    )
    session_id = resp["session_id"]

    try:
        predictor.handle_request(dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        ))

        frame_masks = {}
        for out in predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=session_id)
        ):
            fidx = out.get("frame_index")
            outputs = out.get("outputs", {})
            masks = outputs.get("out_binary_masks")  # (N, H, W) bool ndarray

            if fidx is None or masks is None:
                continue
            if masks.shape[0] == 0:
                continue

            combined = np.any(masks, axis=0)  # (H, W) bool
            # 双向传播时同一帧可能出现两次，后者覆盖前者
            frame_masks[fidx] = combined
    finally:
        predictor.handle_request(
            dict(type="close_session", session_id=session_id)
        )

    # ---------- 读原始视频并应用遮罩 ----------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, quality=8,
                                codec="libx264", pixelformat="yuv420p")

    idx = 0
    total = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1

        if idx in frame_masks:
            mask = frame_masks[idx]
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            masked = frame * mask[:, :, np.newaxis]
        else:
            masked = np.zeros_like(frame)

        writer.append_data(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        idx += 1

    cap.release()
    writer.close()
    return total, len(frame_masks)


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="任务列表 JSON")
    args = ap.parse_args()

    with open(args.tasks, encoding="utf-8") as f:
        tasks = json.load(f)

    if not tasks:
        print("[extract_masks] 没有任务")
        return

    print(f"[extract_masks] 加载 SAM3.1 模型 ...")
    predictor = build_sam3_predictor(version="sam3.1")
    print(f"[extract_masks] 模型就绪，共 {len(tasks)} 个任务\n")

    for i, t in enumerate(tasks):
        vp = t["video_path"]
        prompt = t["sam3_prompt"]
        out = t["output_path"]
        name = os.path.basename(vp)

        print(f"[{i + 1}/{len(tasks)}] {name}  提示: \"{prompt}\"")

        try:
            n_total, n_mask = extract_one(predictor, vp, prompt, out)
            print(f"       {n_total} 帧, {n_mask} 帧有遮罩 → {out}\n")
        except Exception:
            print(f"       [ERROR] 分割失败:")
            traceback.print_exc()
            print()

    print("[extract_masks] 全部完成")


if __name__ == "__main__":
    main()
