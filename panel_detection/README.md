# panel_detection

自包含的 ROS2 面板位姿检测功能包。

基于 YOLOv5 + 深度相机，检测操作面板上的指示灯、旋钮、按钮、螺栓、螺母、阀门、泵共 7 类目标，实时发布 3D 坐标、面板法向量和旋钮角度。

## 平台

| 项目 | 配置 |
|------|------|
| SoC | RK3588 (aarch64) |
| 相机 | 奥比中光 Gemini 336 |
| 推理 | ONNX Runtime CPU (~12 FPS) |
| ROS2 | Humble |

同时保留 Intel RealSense D435i 后端兼容。

## 检测类别与话题

| 话题 | 类别 | 消息类型 | 说明 |
|------|------|----------|------|
| `/panel/info` | 面板整体 | `geometry_msgs/PoseStamped` | 面板中心 + 法向量四元数 |
| `/panel/lights` | light | `std_msgs/String` (JSON) | 指示灯位置 |
| `/panel/buttons` | button | `std_msgs/String` (JSON) | 按钮位置 |
| `/panel/knobs` | knob | `std_msgs/String` (JSON) | 旋钮位置 + 角度 |
| `/panel/bolts` | bolt | `std_msgs/String` (JSON) | 螺栓位置 |
| `/panel/nuts` | nut | `std_msgs/String` (JSON) | 螺母位置 |
| `/panel/valves` | valve | `std_msgs/String` (JSON) | 阀门位置 |
| `/panel/pumps` | pump | `std_msgs/String` (JSON) | 泵位置 |

## 环境要求

- ROS2 Humble / Iron
- Python 3.8+
- OpenCV, NumPy, PyYAML
- ONNX Runtime (CPU 推理) 或 rknn-toolkit-lite2 (NPU 推理)
- pyorbbecsdk (Orbbec 相机) 或 pyrealsense2 (RealSense 相机)

## 准备工作空间

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Tsumugi-W/d435i_detect.git -b ros2-package
```

## 模型转换

在有 PyTorch 环境的机器上执行：

```bash
cd ~/ros2_ws/src/d435i_detect

# 仅导出 ONNX
python panel_detection/scripts/export_model.py \
    --pt newckpt/best.pt \
    --output-dir panel_detection/weights/ \
    --format onnx

# 导出 ONNX + RKNN (需 x86 + rknn-toolkit2)
python panel_detection/scripts/export_model.py \
    --pt newckpt/best.pt \
    --output-dir panel_detection/weights/ \
    --format rknn
```

## 编译

```bash
cd ~/ros2_ws
colcon build --packages-select panel_detection
source install/setup.bash
```

## 运行

```bash
# launch 文件启动 (推荐)
ros2 launch panel_detection panel_detection.launch.py

# 指定自定义配置文件
ros2 launch panel_detection panel_detection.launch.py \
    config_path:=/path/to/your/panel_detection.yaml

# 直接运行节点
ros2 run panel_detection panel_detect_node
```

## 验证话题

```bash
# 查看所有话题
ros2 topic list

# 查看面板位姿
ros2 topic echo /panel/info

# 查看旋钮 (含角度)
ros2 topic echo /panel/knobs

# 查看发布频率
ros2 topic hz /panel/knobs
```

## 话题数据格式

### /panel/info (PoseStamped)

```
header:
  frame_id: "camera_link"
pose:
  position: {x: 0.12, y: -0.03, z: 0.85}
  orientation: {x: 0.01, y: 0.02, z: 0.0, w: 0.99}
```

### /panel/knobs (JSON String)

```json
{
  "stamp": 1716192000.123,
  "knobs": [
    {
      "label": "knob_0",
      "class": "knob",
      "position": {"x": 0.12, "y": -0.05, "z": 0.83},
      "angle": 312.0,
      "confidence": 0.95
    }
  ]
}
```

### /panel/buttons (JSON String)

```json
{
  "stamp": 1716192000.123,
  "buttons": [
    {
      "label": "button_0",
      "class": "button",
      "position": {"x": 0.15, "y": -0.02, "z": 0.82},
      "confidence": 0.91
    }
  ]
}
```

## 旋钮角度估计

支持两种旋钮类型的自动识别：

| 旋钮类型 | 检测方法 | 说明 |
|----------|----------|------|
| 白色指针旋钮 | 灰度二值化 + 轮廓方向 | 黑色旋钮上有白色标记线 |
| 彩色把手旋钮 | 自适应阈值 + 形状方向 | 红色/棕色旋转把手 |

角度以 12 点钟方向为 0 度，顺时针增加，范围 [0, 360)。

## 在机器人端订阅

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json


class PanelSubscriber(Node):
    def __init__(self):
        super().__init__('panel_subscriber')
        self.create_subscription(PoseStamped, '/panel/info', self.info_cb, 10)
        self.create_subscription(String, '/panel/knobs', self.knobs_cb, 10)

    def info_cb(self, msg):
        pos = msg.pose.position
        self.get_logger().info(
            f'面板位置: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})')

    def knobs_cb(self, msg):
        data = json.loads(msg.data)
        for knob in data['knobs']:
            self.get_logger().info(
                f"{knob['label']}: angle={knob.get('angle', '?')} deg, "
                f"pos=({knob['position']['x']:.3f}, "
                f"{knob['position']['y']:.3f}, "
                f"{knob['position']['z']:.3f})")
```

## 配置文件说明

配置文件位于 `config/panel_detection.yaml`，主要配置项：

```yaml
# 相机后端: orbbec / realsense
camera_backend: 'orbbec'

# 推理后端: onnx / rknn / pytorch
inference_backend: 'onnx'

# 模型类别 (与权重一致)
class_name: [ 'light', 'knob', 'bolt', 'nut', 'valve', 'pump', 'button' ]

# 旋钮角度估计开关
knob_angle:
  enable: true
```

## 目录结构

```
panel_detection/
├── package.xml                 # ROS2 包描述
├── setup.py                    # 安装配置
├── setup.cfg
├── config/
│   └── panel_detection.yaml    # 运行配置
├── launch/
│   └── panel_detection.launch.py
├── scripts/
│   └── export_model.py         # 模型转换 (.pt -> .onnx/.rknn)
├── weights/                    # 模型文件
│   └── 0520.onnx
├── resource/
│   └── panel_detection
└── panel_detection/            # Python 包 (自包含所有代码)
    ├── __init__.py
    ├── node_panel_detect.py    # ROS2 检测节点
    ├── camera/                 # 相机抽象层
    │   ├── __init__.py
    │   ├── base.py             # CameraBackend ABC + CameraIntrinsics
    │   ├── orbbec.py           # Orbbec Gemini 336 后端
    │   └── realsense.py        # RealSense D435i 后端
    ├── depth_utils.py          # 深度处理: 反投影、去畸变、滤波、面板法向量
    ├── detector_onnx.py        # ONNX Runtime 推理器
    ├── detector_rknn.py        # RKNN NPU 推理器
    └── knob_angle.py           # 旋钮角度估计 (白色指针 + 彩色把手)
```

## 推理性能

| 后端 | 推理耗时 | FPS | 说明 |
|------|---------|-----|------|
| ONNX Runtime (4线程) | ~80ms | **~12** | 当前默认 |
| RKNN NPU | 预计 ~25ms | 30-48 | 待 NPU 驱动安装 |

## 新设备完整安装教程

以下是在一台全新 RK3588 (Ubuntu 22.04) 设备上从零部署的完整步骤。

### 1. 安装 ROS2 Humble

```bash
sudo apt update && sudo apt install -y ros-humble-ros-base ros-humble-vision-msgs
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. 克隆仓库到 ROS2 工作空间

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone git@github.com:Tsumugi-W/d435i_detect.git -b ros2-package
```

### 3. 创建虚拟环境并安装 Python 依赖

```bash
cd ~/ros2_ws/src/d435i_detect
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install torch==2.7.0 torchvision==0.22.0
pip install pyorbbecsdk2
pip install onnxruntime onnx
pip install opencv-python numpy pyyaml
```

### 4. 配置相机权限 (Orbbec)

```bash
sudo bash $(python3 -c "import pyorbbecsdk,os; print(os.path.dirname(pyorbbecsdk.__file__))")/shared/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger
```

如果使用 RealSense 相机则改为：
```bash
pip install pyrealsense2
# 并将 config/panel_detection.yaml 中 camera_backend 改为 'realsense'
```

### 5. 导出 ONNX 模型 (可选)

如果 `panel_detection/weights/0520.onnx` 已存在则跳过此步。

```bash
cd ~/ros2_ws/src/d435i_detect
PYTHONPATH=. python panel_detection/scripts/export_model.py \
    --pt weights/0520.pt \
    --output-dir panel_detection/weights/ \
    --format onnx
```

### 6. 编译功能包

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source src/d435i_detect/.venv/bin/activate
colcon build --packages-select panel_detection
source install/setup.bash
```

### 7. 运行

```bash
# 推荐: 通过 launch 文件启动
ros2 launch panel_detection panel_detection.launch.py

# 或直接运行节点
ros2 run panel_detection panel_detect_node
```

### 8. 验证

```bash
# 另开终端
source ~/ros2_ws/install/setup.bash
ros2 topic list            # 应看到 /panel/info, /panel/knobs 等话题
ros2 topic echo /panel/knobs
```

### 日常使用

每次新开终端需要：
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/src/d435i_detect/.venv/bin/activate
source ~/ros2_ws/install/setup.bash
```

建议写入 `~/.bashrc`：
```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
echo "source ~/ros2_ws/src/d435i_detect/.venv/bin/activate" >> ~/.bashrc
```

## 许可证

Apache 2.0
