# YOLOv5 3D Detection — RK3588 + Orbbec Gemini 336

**基于 YOLOv5 的实时 3D 目标检测系统，支持 ROS2 消息发布。**

检测操作面板上的旋钮/指示灯，输出 3D 坐标和面板法向量。

## 平台

| 项目 | 配置 |
|------|------|
| SoC | RK3588 (aarch64) |
| 相机 | 奥比中光 Gemini 336 |
| 推理 | ONNX Runtime CPU (~12 FPS) |
| ROS2 | Humble |

同时保留 Intel RealSense D435i 后端兼容。

## 快速开始

```bash
# 1. 激活环境
cd /home/ztl/project/d435i_detect
source .venv/bin/activate

# 2. 一键检测（不依赖 ROS2）
python run.py

# 3. ROS2 版（面板法向量 + 话题发布）
source /opt/ros/humble/setup.bash
python rstest3.py
```

## 脚本说明

| 脚本 | 功能 | 依赖 ROS2 |
|------|------|-----------|
| `run.py` | 一键检测 + 3D 坐标 + 面板法向量 + 深度可视化 | 否 |
| `rstest3.py` | 完整版：检测 + 3D + 面板法向量 + ROS2 发布 | 是 |
| `record.py` | 相机录制，采集 YOLO 训练数据 | 否 |

### run.py 命令行参数

```bash
python run.py                        # 默认 best.pt + ONNX 推理
python run.py --weight best.pt       # 指定模型
python run.py --backend pytorch      # 切回 PyTorch 推理
python run.py --backend onnx         # ONNX Runtime（默认）
python run.py --camera realsense     # 切换相机后端
python run.py --no-depth             # 只检测不算 3D
```

### record.py 录制数据

```bash
python record.py                     # 启动录制
python record.py --save-depth        # 同时录深度
python record.py --snap-only         # 只截图不录视频
python record.py --res 1280 720      # 高分辨率
# 快捷键: s=截图  r=暂停/继续  q=退出
```

### ROS2 话题

```bash
# rstest3.py 发布的话题
ros2 topic echo /detection_3d        # Detection3DArray: 3D坐标 + 法向量四元数
ros2 topic echo /detection_coords    # String: {'xyz': [...], 'panel_normal': [...]}
```

## 配置文件

`config/yolov5s.yaml`:

```yaml
# 相机
camera_backend: 'orbbec'       # 'orbbec' | 'realsense'
camera:
  color_width: 640
  color_height: 480
  fps: 30

# 推理
inference_backend: 'onnx'      # 'onnx' | 'pytorch' | 'rknn'
onnx_threads: 4

# 模型
weight: "weights/yolov5s.pt"
input_size: 640
class_num: 2
class_name: ['knob', 'indicator_light_on']
threshold:
  confidence: 0.3
  iou: 0.01
device: 'cpu'
```

## 项目结构

```
.
├── run.py                   # 一键检测（推荐入口）
├── rstest3.py               # ROS2 版完整检测
├── record.py                # 数据录制
├── best.pt                  # 自训练模型权重
├── best.onnx                # ONNX 格式模型（自动导出）
├── config/
│   └── yolov5s.yaml         # 统一配置文件
├── camera/                  # 相机抽象层
│   ├── base.py              # CameraBackend ABC + CameraIntrinsics
│   ├── orbbec.py            # Orbbec Gemini 336 后端
│   └── realsense.py         # RealSense D435i 后端
├── depth_utils.py           # 深度处理：反投影、去畸变、滤波、面板法向量
├── detector_onnx.py         # ONNX Runtime 推理器
├── detector_rknn.py         # RKNN NPU 推理器（待 NPU 驱动）
├── app_config.py            # 配置加载 + 后端工厂
├── models/                  # YOLOv5 模型定义
├── utils/                   # YOLOv5 工具函数
├── tools/
│   └── convert_to_rknn.py   # PT→ONNX→RKNN 模型转换
├── requirements.txt         # Python 依赖
└── requirements_rk3588.txt  # RK3588 平台依赖说明
```

## 推理性能

| 后端 | 推理耗时 | FPS | 说明 |
|------|---------|-----|------|
| PyTorch CPU | ~340ms | ~3 | 未优化基线 |
| ONNX Runtime (4线程) | ~80ms | **~12** | 当前默认 |
| RKNN NPU | 预计 ~25ms | 30-48 | 待 NPU 驱动安装 |

优化措施：ONNX Runtime 推理 + 异步取帧（独立线程，不阻塞推理）。

## 精度优化

- **多帧融合** — 5 帧加权平均，20-30% 改善
- **深度双边滤波** — 平滑深度噪声保留边界，10-15% 改善
- **多点采样** — 7x7 中位数，10-15% 改善
- **去畸变** — Brown-Conrady 模型迭代反演
- **面板法向量** — RANSAC+SVD 拟合操作面板平面，所有目标共用

## 安装

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 安装依赖
pip install torch==2.7.0 torchvision==0.22.0
pip install pyorbbecsdk2
pip install onnxruntime onnx
pip install -r requirements.txt

# Orbbec 相机 udev 权限
sudo bash $(python3 -c "import pyorbbecsdk,os; print(os.path.dirname(pyorbbecsdk.__file__))")/shared/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger

# ROS2（可选）
sudo apt install ros-humble-ros-base ros-humble-vision-msgs
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## 许可证

Apache 2.0

---

**最后更新**：2026-04-15
**版本**：4.0 (RK3588 + Orbbec Gemini 336)
