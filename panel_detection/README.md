# panel_detection

基于 YOLOv5 + 深度相机的 ROS2 面板位姿检测功能包。

检测操作面板上的旋钮、按钮、螺栓、螺母、阀门、泵共 6 类目标，实时发布 3D 坐标和面板法向量。支持 ONNX Runtime CPU 推理和 RK3588 NPU 加速。

## 检测类别与话题

| 话题 | 类别 | 消息类型 | 说明 |
|------|------|----------|------|
| `/panel/info` | 面板整体 | `geometry_msgs/PoseStamped` | 面板中心 + 法向量四元数 |
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
- ONNX Runtime（CPU 推理）或 rknn-toolkit-lite2（NPU 推理）
- pyorbbecsdk（Orbbec 相机）或 pyrealsense2（RealSense 相机）

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

# 导出 ONNX + RKNN（需 x86 + rknn-toolkit2）
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
# launch 文件启动（推荐）
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

# 查看旋钮（含角度）
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
# 相机后端：orbbec / realsense
camera_backend: 'orbbec'

# 推理后端：onnx / rknn / pytorch
inference_backend: 'onnx'

# 模型类别（与权重一致）
class_name: [ 'button', 'knob', 'bolt', 'nut', 'valve', 'pump' ]

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
├── weights/                    # 放置转换后的模型文件
├── resource/
│   └── panel_detection
└── panel_detection/
    ├── __init__.py
    └── node_panel_detect.py    # 核心检测节点
```

节点运行时复用项目根目录的 `camera/`、`depth_utils.py`、`knob_angle.py`、`detector_onnx.py` 等模块，无需复制代码。
