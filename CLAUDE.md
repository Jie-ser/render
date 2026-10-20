# Render Project

本项目用于机器人仿真到真实视频的渲染 (sim-to-real)。包含三个子项目，协同工作完成从 RGB 视频到逼真机器人视频的生成。

## 整体流程

```
原始 RGB 视频 (robot_demo.mp4)
    ├── Video-Depth-Anything  →  深度视频 (Depth MP4)
    └── SAM3 / 视频分割       →  机器人遮罩视频 (Robot Mask MP4)
                                    ↓
              RoboRender-Video-Model (深度 + 遮罩 + 文本提示 → 真实感 RGB 视频)
```

## 子项目概览

### 1. Video-Depth-Anything (`Video-Depth-Anything/`)

**功能**: 视频深度估计 (CVPR 2025, ByteDance)，对任意长度视频生成逐像素深度图，保持时间一致性。

**运行方式**:
```bash
python run.py --input_video <video.mp4> --output_dir ./outputs --encoder vitl
python run.py --input_video <video.mp4> --output_dir ./outputs --encoder vitl --metric  # 度量深度
```

**关键参数**:
- `--encoder`: 模型大小 — `vits`(28M) / `vitb`(113M) / `vitl`(382M)
- `--input_size`: 推理分辨率，默认 518
- `--max_res`: 最大分辨率，默认 1280
- `--grayscale`: 输出灰度深度而非彩色
- `--save_npz` / `--save_exr`: 保存原始深度数据

**输入**: 任意 MP4 视频文件
**输出**: `{name}_src.mp4` (源视频), `{name}_vis.mp4` (深度可视化), 可选 `.npz` / `.exr` 原始深度

**模型权重**: 放在 `checkpoints/` 目录下
- `video_depth_anything_vit{s,b,l}.pth` (相对深度)
- `metric_video_depth_anything_vit{s,b,l}.pth` (度量深度)
- 来源: HuggingFace `depth-anything/`

**依赖**: Python, PyTorch 2.1.1, CUDA, xformers, decord, OpenEXR

---

### 2. SAM3 (`sam3/`)

**功能**: Segment Anything Model 3 (Meta), 统一的图像/视频分割基础模型，支持文本、点、框、遮罩提示。848M 参数。

**运行方式 (视频分割)**:
```python
from sam3.model_builder import build_sam3_predictor

predictor = build_sam3_predictor(version="sam3.1", compile=True)

# 启动会话
resp = predictor.handle_request({"type": "start_session", "resource_path": "<video_path>"})
session_id = resp["session_id"]

# 添加文本提示 (例如分割机器人)
predictor.handle_request({
    "type": "add_prompt", "session_id": session_id,
    "frame_index": 0, "text": "robot arm"
})

# 传播到整个视频
for out in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
    masks = out["out_binary_masks"]  # 每帧的二值遮罩

predictor.handle_request({"type": "close_session", "session_id": session_id})
```

**输入**: JPEG 帧目录 / MP4 视频 / 单张图片 / PIL Image 列表
**输出**: 每帧每对象的二值分割遮罩、检测框、置信分数

**模型权重**: SAM3.1 multiplex checkpoint，推荐使用已通过 ModelScope 下载到服务器的本地文件 `sam3.1_multiplex.pt`。
- 服务器权重目录：`/pfs/mayuema/spj/render/sam3/checkpoints/`，SAM3.1 视频分割使用 `/pfs/mayuema/spj/render/sam3/checkpoints/sam3.1_multiplex.pt`（SAM3.1 multiplex，约 3.26 GB）；同目录也有 `sam3.pt`（基础 SAM3，约 3.21 GB）。
- `run_pipeline.py` 的遮罩步骤必须显式提供本地 checkpoint：在 `pipeline_config.json` 根对象设置 `sam3_checkpoint`，或用 `--sam3-checkpoint` / `SAM3_CHECKPOINT` 环境变量设置。
- 默认使用上述服务器上的 SAM3.1 checkpoint；命令行、配置文件或环境变量可以覆盖默认值。
- `scripts/extract_masks.py` 会验证该文件存在并通过 `checkpoint_path` 加载；文件无效时立即报错，不会访问 Hugging Face 或自动下载。
- SAM3 与 SAM3.1 的 `init_state()` 参数并不完全相同；共享 predictor 只会向模型转发其 `init_state()` 签名支持的参数，避免 SAM3.1 因 `offload_state_to_cpu` 参数不兼容而在每个视频会话启动时失败。
- SAM3.1 推理默认关闭 FlashAttention 3，使用 PyTorch SDPA；FA3 kernel 面向 Hopper GPU，其他架构可能在文本提示推理时报 `no kernel image is available`。仅在已确认 GPU 与 FA3 kernel 兼容时才向 `extract_masks.py` 显式传 `--use-fa3`。
- 遮罩任务发生异常时，`extract_masks.py` 汇总失败视频并返回非零退出码；pipeline 错误会同时显示首个 traceback 提示和未生成文件清单。

**依赖**: Python >= 3.12, PyTorch >= 2.7, CUDA >= 12.6, timm, huggingface_hub

---

### 3. RoboRender-Video-Model (`RoboRender-Video-Model/`)

**功能**: 仿真到真实的视频转换系统，基于 Wan2.1-Fun-V1.1-1.3B-Control 扩散模型 + 自定义 LoRA。接收深度视频 + 机器人遮罩 + 文本提示，生成逼真 RGB 视频。支持多视角 (hand_left, hand_right, head_color)。

**运行方式**:
```bash
python infer.py --sample_dir data/test_sample \
    --views hand_left hand_right head_color \
    --output_dir outputs/demo
```

**输入数据目录结构**:
```
sample/
  original_instruction.txt              # 文本提示
  <view>/depth/original_depth_vda.mp4   # 深度控制视频 (必需)
  <view>/mask/rgb_robot.mp4             # 机器人遮罩视频 (可选)
  <view>/pre_frame.png                  # 参考帧 (可选)
```

**关键参数**:
- `--views`: 视角名称, 如 `hand_left hand_right head_color`
- `--height / --width`: 每视角分辨率，默认 240x416
- `--clip_length`: 每片段帧数，默认 81
- `--num_inference_steps`: 扩散步数，默认 50
- `--use_prev_frame`: 使用前一帧作为参考
- `--teacache`: 启用 TeaCache 加速 (~2.5x)

**输出**: `generated_<view>.mp4` (每视角生成视频), `comparison.mp4` (对比视频)

**模型权重**:
- `models/Wan2.1-Fun-V1.1-1.3B-Control/` — 来自 `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control`
- `ckpt/roborender_lora.safetensors` — 来自 `RoboRender/roborender-video`

**依赖**: Python 3.10, PyTorch 2.9+, CUDA 12.8, transformers, safetensors, ~20GB 显存 (三视角)

---

## 数据准备流程 (RGB 视频 → RoboRender 输入)

1. **生成深度视频**: 使用 Video-Depth-Anything 对 RGB 视频进行深度估计
   ```bash
   python Video-Depth-Anything/run.py --input_video <rgb.mp4> --output_dir ./depth_output --encoder vitl
   ```
   输出的 `*_vis.mp4` 即为深度视频

2. **生成机器人遮罩**: 使用 SAM3 对 RGB 视频进行机器人分割，导出每帧二值遮罩，合成为遮罩视频

3. **组织 RoboRender 输入目录**: 按上述目录结构放置深度视频和遮罩视频

4. **运行 RoboRender**: 生成逼真 RGB 视频

## 开发工作流

- 本地修改代码 → `git push` 推送到 GitHub → 服务器 `git pull` 拉取更新
- 仓库远程地址: `https://github.com/Jie-ser/render.git`
- 所有提交仅以仓库所有者 (Jie-ser) 名义进行，不引入任何 Co-Author
- 提交信息使用中文或英文均可

## 服务器部署信息

### 硬件环境
- GPU: 4x NVIDIA RPBZZZ6 (~98GB VRAM each)
- CUDA Driver: 580.126.20, CUDA Version: 13.0
- 代理: `http://10.184.64.107:18000`

### 目录结构
```
/pfs/mayuema/spj/render/               ← 项目根目录 (git clone)
├── Video-Depth-Anything/
│   └── checkpoints/                   ← VDA 模型权重
│       ├── video_depth_anything_vits.pth
│       └── video_depth_anything_vitl.pth
├── sam3/
│   ├── checkpoints/
│   │   ├── sam3.pt
│   │   └── sam3.1_multiplex.pt       ← SAM3.1 pipeline 默认权重
│   └── (SAM3源码)
├── RoboRender-Video-Model/
│   ├── models/Wan2.1-Fun-V1.1-1.3B-Control/  ← 基础模型权重
│   └── ckpt/roborender_lora.safetensors       ← LoRA 权重
└── .hf_cache/                         ← 其他 HuggingFace 模型的缓存目录
```

### Conda 环境
| 环境名 | Python | PyTorch | CUDA | 用途 |
|--------|--------|---------|------|------|
| `vda` | 3.10 | 2.1.1 | cu121 | Video-Depth-Anything |
| `sam3` | 3.12 | 2.10.0 | cu128 | SAM3 视频分割 |
| `roborender` | 3.10 | 2.9.1 | cu128 | RoboRender 推理 |

### 服务器常用命令
```bash
# 设置代理 (每次开新终端都需要)
export http_proxy=http://10.184.64.107:18000
export https_proxy=http://10.184.64.107:18000

# 拉取最新代码
cd /pfs/mayuema/spj/render && git pull

# 激活环境
conda activate vda          # Video-Depth-Anything
conda activate sam3         # SAM3
conda activate roborender   # RoboRender
```
