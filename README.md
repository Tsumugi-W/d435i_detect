# YOLOv5 3D Detection — RK3588 + Orbbec Gemini 336

**基于 YOLOv5 的实时 3D 目标检测系统，支持旋钮角度估计和 ROS2 消息发布。**

检测操作面板上的旋钮/指示灯，输出 3D 坐标、面板法向量和旋钮旋转角度。

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

# 3. ROS2 版（话题发布）
source /opt/ros/humble/setup.bash
python rstest3.py
```

## 脚本说明

| 脚本 | 功能 | 依赖 ROS2 |
|------|------|-----------|
| `run.py` | 一键检测 + 3D 坐标 + 面板法向量 + 旋钮角度 + 深度可视化 | 否 |
| `rstest3.py` | 完整版：检测 + 3D + 法向量 + 角度 + ROS2 话题发布 | 是 |
| `record.py` | 相机录制，采集 YOLO 训练数据 | 否 |
| `tools/test_video.py` | 离线视频测试，验证检测+角度效果，生成标注视频 | 否 |

### run.py

```bash
python run.py                        # 默认 best.pt + ONNX 推理
python run.py --weight best.pt       # 指定模型
python run.py --backend pytorch      # 切回 PyTorch 推理
python run.py --backend onnx         # ONNX Runtime（默认）
python run.py --camera realsense     # 切换相机后端
python run.py --no-depth             # 只检测不算 3D
```

### record.py

```bash
python record.py                     # 启动录制
python record.py --save-depth        # 同时录深度
python record.py --snap-only         # 只截图不录视频
python record.py --res 1280 720      # 高分辨率
# 快捷键: s=截图  r=暂停/继续  q=退出
```

### tools/test_video.py

```bash
python tools/test_video.py                          # 自动找最新录制
python tools/test_video.py -i recordings/xxx/color.mp4 --save   # 指定视频，保存结果
python tools/test_video.py --no-show --max-frames 500           # 无 GUI 纯跑数
python tools/test_video.py --skip 400 --max-frames 800 --save   # 跳到有效区间
# 显示模式下: 空格=暂停  q/ESC=退出
```

### ROS2 话题（rstest3.py）

```bash
ros2 topic echo /panel/info      # PoseStamped: 面板中心坐标 + 法向量四元数
ros2 topic echo /panel/knobs     # JSON: 旋钮位置 + 角度 + 标签
ros2 topic echo /panel/buttons   # JSON: 按钮位置 + 标签
```

**`/panel/info` 消息格式 (PoseStamped):**
```
header:
  stamp: {sec: ..., nanosec: ...}
  frame_id: "camera_link"
pose:
  position:                    # 面板中心 3D 坐标（米）
    x: 0.12
    y: -0.03
    z: 0.85
  orientation:                 # 面板法向量四元数（末端执行器进近方向）
    x: 0.01
    y: 0.02
    z: -0.01
    w: 0.99
```
每 10 帧更新一次并缓存，未检出目标时 position 为零、orientation 为单位四元数。

**`/panel/knobs` 消息格式:**
```json
{
  "stamp": 1234567890.123,
  "knobs": [
    {"label": "knob_0", "position": {"x": 0.12, "y": -0.05, "z": 0.83}, "angle": 312.0, "confidence": 0.95}
  ]
}
```

**`/panel/buttons` 消息格式:**
```json
{
  "stamp": 1234567890.123,
  "buttons": [
    {"label": "button_0", "position": {"x": 0.15, "y": -0.02, "z": 0.82}, "confidence": 0.91}
  ]
}
```

目标按空间位置排序（上到下、左到右），标签 `knob_0/1/2...` 跨帧稳定。

## 旋钮角度估计

基于传统 CV，无需额外模型：
1. 自适应二值化（局部高斯 + OTSU + 固定阈值，自动选最佳）
2. 形态学去噪
3. 轮廓筛选（面积 + 长宽比）
4. fitLine 拟合方向 + 质心消解 180 度歧义

以 12 点钟方向为 0 度，顺时针增加，范围 [0, 360)。

**离线测试结果:**

| 视频 | 检出帧率 | 旋钮数 | 角度成功率 |
|------|---------|--------|-----------|
| 173951 | 62.9% | 1432 | 86.1% |
| 175403 | 78.2% | 1943 | 90.1% |

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

# 旋钮角度
knob_angle:
  enable: true
  binary_thresh: 180
  circle_mask_ratio: 0.85
  knob_class: 'knob'

# ROS2 话题
ros2_topics:
  panel_info: '/panel/info'
  knobs: '/panel/knobs'
  buttons: '/panel/buttons'
```

## 项目结构

```
.
├── run.py                   # 一键检测（推荐入口）
├── rstest3.py               # ROS2 版完整检测
├── record.py                # 数据录制
├── best.pt / best.onnx      # 模型权重
├── knob_angle.py            # 旋钮角度估计模块
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
├── tools/
│   ├── test_video.py        # 离线视频测试 + 效果验证
│   └── convert_to_rknn.py   # PT→ONNX→RKNN 模型转换
├── models/                  # YOLOv5 模型定义
├── utils/                   # YOLOv5 工具函数
├── requirements.txt         # Python 依赖
└── requirements_rk3588.txt  # RK3588 平台依赖说明
```

## 推理性能

| 后端 | 推理耗时 | FPS | 说明 |
|------|---------|-----|------|
| PyTorch CPU | ~340ms | ~3 | 未优化基线 |
| ONNX Runtime (4线程) | ~80ms | **~12** | 当前默认 |
| RKNN NPU | 预计 ~25ms | 30-48 | 待 NPU 驱动安装 |

优化措施：ONNX Runtime 推理 + 异步取帧（独立线程深拷贝，不阻塞推理）。

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

**最后更新**: 2026-04-17
**版本**: 4.1 (旋钮角度估计 + ROS2 话题改造)
