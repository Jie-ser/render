#!/usr/bin/env python3
"""
RoboRender 全自动推理流水线

用法:
    python run_pipeline.py --config pipeline_config.json --gpu 1
    python run_pipeline.py --config pipeline_config.json --gpu 2 --encoder vits --skip preprocess depth

流程: 预处理(ffmpeg) → 深度估计(VDA) → 机器人分割(SAM3) → 数据组织 → RoboRender推理
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VDA_DIR = os.path.join(PROJECT_ROOT, "Video-Depth-Anything")
SAM3_DIR = os.path.join(PROJECT_ROOT, "sam3")
ROBORENDER_DIR = os.path.join(PROJECT_ROOT, "RoboRender-Video-Model")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# Server-side conda environments.  Activate by absolute prefix so the
# pipeline does not depend on the caller's environment names or cwd.
ENV_PATHS = {
    "vda": "/pfs/mayuema/depth/env/vda",
    "sam3": "/pfs/mayuema/depth/env/sam3",
    "roborender": "/pfs/mayuema/depth/env/roborender",
}


# ======================================================================
# 工具函数
# ======================================================================

def log(msg, level="INFO"):
    tag = {"INFO": "   ", "OK": " ✓ ", "WARN": " ⚠ ", "ERR": " ✗ ", "RUN": " ▶ "}
    print(f"{tag.get(level, '   ')}{msg}", flush=True)


def get_conda_base():
    r = subprocess.run("conda info --base", shell=True,
                       capture_output=True, text=True)
    base = r.stdout.strip()
    if not base:
        log("无法获取 conda 路径，请确保 conda 已初始化", "ERR")
        sys.exit(1)
    return base


CONDA_BASE = None


def run_in_env(env_name, cmd, cwd=None, gpu_id="0"):
    """在指定 conda 环境中执行 shell 命令"""
    global CONDA_BASE
    if env_name not in ENV_PATHS:
        raise ValueError(f"未知 conda 环境: {env_name}")
    if CONDA_BASE is None:
        CONDA_BASE = get_conda_base()
    env_path = ENV_PATHS[env_name]
    full = (
        f'bash -c "'
        f"source {CONDA_BASE}/etc/profile.d/conda.sh && "
        f"conda activate {env_path} && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} {cmd}"
        f'"'
    )
    log(f"[{env_name} @ {env_path}] {cmd}", "RUN")
    return subprocess.run(full, shell=True, cwd=cwd).returncode


def paths_for(sample_name, view_name, work_dir):
    """确定性计算所有中间/输出路径"""
    return {
        "prep": os.path.join(work_dir, "preprocessed", sample_name,
                             f"{view_name}.mp4"),
        "depth_dir": os.path.join(work_dir, "depth", sample_name, view_name),
        "depth_vis": os.path.join(work_dir, "depth", sample_name, view_name,
                                  f"{view_name}_vis.mp4"),
        "mask": os.path.join(work_dir, "masks", sample_name,
                             f"{view_name}_mask.mp4"),
        "rr_depth": os.path.join(work_dir, "roborender_input", sample_name,
                                 view_name, "depth", "original_depth_vda.mp4"),
        "rr_mask": os.path.join(work_dir, "roborender_input", sample_name,
                                view_name, "mask", "rgb_robot.mp4"),
        "rr_sample": os.path.join(work_dir, "roborender_input", sample_name),
    }


# ======================================================================
# Step 1: 视频预处理
# ======================================================================

def step_preprocess(config, work_dir, width, height, fps):
    print(f"\n{'='*60}")
    print(f"  Step 1/5  视频预处理 → {width}x{height} @{fps}fps")
    print(f"{'='*60}")

    ok = subprocess.run("ffmpeg -version", shell=True,
                        capture_output=True).returncode == 0
    if not ok:
        log("ffmpeg 未安装，无法预处理", "ERR")
        return

    for sample in config["samples"]:
        for view in sample["views"]:
            src = view["video_path"]
            p = paths_for(sample["name"], view["view_name"], work_dir)
            dst = p["prep"]

            if not os.path.isfile(src):
                log(f"文件不存在: {src}", "ERR")
                continue

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cmd = [
                "ffmpeg", "-y", "-i", src,
                "-vf",
                f"scale={width}:{height}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                "-r", str(fps),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                dst,
            ]
            log(f"{os.path.basename(src)} → {os.path.basename(dst)}")
            ret = subprocess.run(cmd, capture_output=True)
            if ret.returncode == 0:
                log(f"完成: {dst}", "OK")
            else:
                stderr = ret.stderr.decode(errors="replace")[-300:]
                log(f"ffmpeg 错误: {stderr}", "ERR")


# ======================================================================
# Step 2: 深度估计 (Video-Depth-Anything)
# ======================================================================

def step_depth(config, work_dir, gpu_id, encoder):
    print(f"\n{'='*60}")
    print(f"  Step 2/5  深度估计 (Video-Depth-Anything, encoder={encoder})")
    print(f"{'='*60}")

    for sample in config["samples"]:
        for view in sample["views"]:
            p = paths_for(sample["name"], view["view_name"], work_dir)
            src = p["prep"]

            if not os.path.isfile(src):
                log(f"预处理视频缺失，跳过: {src}", "WARN")
                continue

            os.makedirs(p["depth_dir"], exist_ok=True)
            log(f"{sample['name']}/{view['view_name']}")
            cmd = (f"python run.py "
                   f"--input_video {src} "
                   f"--output_dir {p['depth_dir']} "
                   f"--encoder {encoder}")
            ret = run_in_env("vda", cmd, cwd=VDA_DIR, gpu_id=gpu_id)

            if ret == 0 and os.path.isfile(p["depth_vis"]):
                log(f"完成: {p['depth_vis']}", "OK")
            else:
                log(f"深度估计失败 (exit {ret})", "ERR")


# ======================================================================
# Step 3: 机器人分割 (SAM3)
# ======================================================================

def step_mask(config, work_dir, gpu_id):
    print(f"\n{'='*60}")
    print(f"  Step 3/5  机器人分割 (SAM3)")
    print(f"{'='*60}")

    tasks = []
    for sample in config["samples"]:
        for view in sample["views"]:
            p = paths_for(sample["name"], view["view_name"], work_dir)
            if not os.path.isfile(p["prep"]):
                log(f"预处理视频缺失，跳过: {p['prep']}", "WARN")
                continue
            tasks.append({
                "video_path": p["prep"],
                "sam3_prompt": view["sam3_prompt"],
                "output_path": p["mask"],
            })

    if not tasks:
        log("没有需要分割的视频", "WARN")
        return

    tasks_file = os.path.join(work_dir, "_sam3_tasks.json")
    with open(tasks_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    log(f"共 {len(tasks)} 个分割任务")
    script = os.path.join(SCRIPTS_DIR, "extract_masks.py")
    cmd = f"python {script} --tasks {tasks_file}"
    ret = run_in_env("sam3", cmd, cwd=SAM3_DIR, gpu_id=gpu_id)

    if ret != 0:
        log(f"SAM3 返回错误 (exit {ret})", "ERR")

    for t in tasks:
        if os.path.isfile(t["output_path"]):
            log(f"完成: {os.path.basename(t['output_path'])}", "OK")
        else:
            log(f"遮罩未生成: {t['output_path']}", "WARN")


# ======================================================================
# Step 4: 组织 RoboRender 输入目录
# ======================================================================

def step_organize(config, work_dir):
    print(f"\n{'='*60}")
    print(f"  Step 4/5  组织 RoboRender 输入数据")
    print(f"{'='*60}")

    for sample in config["samples"]:
        p0 = paths_for(sample["name"], sample["views"][0]["view_name"], work_dir)
        sample_dir = p0["rr_sample"]
        os.makedirs(sample_dir, exist_ok=True)

        prompt_file = os.path.join(sample_dir, "original_instruction.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(sample["roborender_prompt"])
        log(f"{sample['name']}: prompt 已写入")

        for view in sample["views"]:
            p = paths_for(sample["name"], view["view_name"], work_dir)

            # Optional first-frame reference for RoboRender clip 0.  The
            # case manifest stores these as <view>_frame0.png; RoboRender
            # expects the canonical name pre_frame.png inside each view.
            pre_frame_src = view.get("pre_frame_path")
            if pre_frame_src and os.path.isfile(pre_frame_src):
                pre_frame_dst = os.path.join(
                    p["rr_sample"], view["view_name"], "pre_frame.png"
                )
                os.makedirs(os.path.dirname(pre_frame_dst), exist_ok=True)
                shutil.copy2(pre_frame_src, pre_frame_dst)
                log(f"  {view['view_name']}/pre_frame.png ← 参考帧", "OK")

            os.makedirs(os.path.dirname(p["rr_depth"]), exist_ok=True)
            if os.path.isfile(p["depth_vis"]):
                shutil.copy2(p["depth_vis"], p["rr_depth"])
                log(f"  {view['view_name']}/depth ← 深度视频", "OK")
            else:
                log(f"  {view['view_name']}/depth 缺失", "WARN")

            os.makedirs(os.path.dirname(p["rr_mask"]), exist_ok=True)
            if os.path.isfile(p["mask"]):
                shutil.copy2(p["mask"], p["rr_mask"])
                log(f"  {view['view_name']}/mask  ← 遮罩视频", "OK")
            else:
                log(f"  {view['view_name']}/mask  缺失 (RoboRender 会跳过遮罩)", "WARN")


# ======================================================================
# Step 5: RoboRender 推理
# ======================================================================

def step_roborender(config, work_dir, gpu_id):
    print(f"\n{'='*60}")
    print(f"  Step 5/5  RoboRender 推理")
    print(f"{'='*60}")

    gen_dir = os.path.join(work_dir, "generated")

    for sample in config["samples"]:
        p0 = paths_for(sample["name"], sample["views"][0]["view_name"], work_dir)
        sample_dir = p0["rr_sample"]

        if not os.path.isdir(sample_dir):
            log(f"跳过 {sample['name']} (输入目录不存在)", "WARN")
            continue

        views = " ".join(v["view_name"] for v in sample["views"])
        out = os.path.join(gen_dir, sample["name"])
        os.makedirs(out, exist_ok=True)

        # Use any available real first frames for clip-0 conditioning. Views
        # without a supplied frame are left unconditioned by infer.py.
        ref_specs = []
        for view in sample["views"]:
            ref_path = os.path.join(
                sample_dir, view["view_name"], "pre_frame.png"
            )
            if os.path.isfile(ref_path):
                ref_specs.append(f"{view['view_name']}:{ref_path}")

        log(f"{sample['name']} (视角: {views})")
        cmd = (f"python infer.py "
               f"--sample_dir {sample_dir} "
               f"--views {views} "
               f"--output_dir {out}")
        if ref_specs:
            cmd += " --first_clip_refs " + " ".join(ref_specs)
        ret = run_in_env("roborender", cmd, cwd=ROBORENDER_DIR, gpu_id=gpu_id)

        if ret == 0:
            log(f"完成 → {out}", "OK")
        else:
            log(f"RoboRender 失败 (exit {ret})", "ERR")


# ======================================================================
# 主入口
# ======================================================================

def main():
    ap = argparse.ArgumentParser(
        description="RoboRender 全自动推理流水线",
        epilog="示例: python run_pipeline.py --config pipeline_config.json --gpu 1",
    )
    ap.add_argument("--config", required=True, help="配置文件 (JSON)")
    ap.add_argument("--gpu", default="1", help="GPU 编号 (默认: 1)")
    ap.add_argument("--output_dir", default="./pipeline_output", help="输出根目录")
    ap.add_argument("--encoder", default="vitl",
                    choices=["vits", "vitb", "vitl"], help="VDA 编码器")
    ap.add_argument("--width", type=int, default=416)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["preprocess", "depth", "mask", "organize",
                             "roborender"],
                    help="跳过指定步骤")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)

    n_samples = len(config["samples"])
    n_videos = sum(len(s["views"]) for s in config["samples"])

    print(f"\n{'='*60}")
    print(f"  RoboRender 流水线")
    print(f"  样本: {n_samples}  视频: {n_videos}  GPU: {args.gpu}")
    print(f"  输出: {work_dir}")
    if args.skip:
        print(f"  跳过: {', '.join(args.skip)}")
    print(f"{'='*60}")

    steps = [
        ("preprocess", lambda: step_preprocess(config, work_dir,
                                               args.width, args.height,
                                               args.fps)),
        ("depth",      lambda: step_depth(config, work_dir,
                                          args.gpu, args.encoder)),
        ("mask",       lambda: step_mask(config, work_dir, args.gpu)),
        ("organize",   lambda: step_organize(config, work_dir)),
        ("roborender", lambda: step_roborender(config, work_dir, args.gpu)),
    ]

    for name, fn in steps:
        if name in args.skip:
            print(f"\n  [跳过] {name}")
            continue
        fn()

    result_file = os.path.join(work_dir, "pipeline_result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  流水线完成!")
    print(f"  生成结果: {os.path.join(work_dir, 'generated')}")
    print(f"  运行记录: {result_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
