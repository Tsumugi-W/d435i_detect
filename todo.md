# ROS2 话题改造计划

## 目标

将当前混合式话题结构改为按功能分离、语义明确的结构，方便机器人端直接使用。

## 当前状态

- `/detection_3d` (Detection3DArray) — **已移除**
- `/detection_coords` (String) — **已移除**

## 改造后话题

```
/panel/info      geometry_msgs/PoseStamped   面板位姿（中心点 + 法向量四元数）
/panel/knobs     String (JSON)               所有旋钮：位置 + 角度 + 标识
/panel/buttons   String (JSON)               所有按钮：位置 + 状态 + 标识
```

> 如果后续引入 ROS2 包结构（colcon build），可将 knobs/buttons 替换为自定义 msg。
> 当前用 JSON String 最简单，无需编译。

### 消息格式定义

`/panel/info`:
```
直接使用 geometry_msgs/PoseStamped
- header.frame_id = "camera_link"
- pose.position = 面板中心 3D 坐标
- pose.orientation = 法向量四元数（即末端执行器进近方向）
- 发布频率：每 10 帧更新一次
```

`/panel/knobs` JSON 格式:
```json
{
  "stamp": 1234567890.123,
  "knobs": [
    {
      "label": "knob_0",
      "position": {"x": 0.12, "y": -0.05, "z": 0.83},
      "angle": 312.0,
      "confidence": 0.95
    }
  ]
}
```

`/panel/buttons` JSON 格式:
```json
{
  "stamp": 1234567890.123,
  "buttons": [
    {
      "label": "button_0",
      "position": {"x": 0.15, "y": -0.02, "z": 0.82},
      "confidence": 0.91
    }
  ]
}
```

---

## 改造清单

### 1. rstest3.py — 发布逻辑重写

- [x] 新建 3 个 publisher 替换现有 2 个
  - `self.panel_info_pub = create_publisher(PoseStamped, '/panel/info', 10)`
  - `self.knobs_pub = create_publisher(String, '/panel/knobs', 10)`
  - `self.buttons_pub = create_publisher(String, '/panel/buttons', 10)`
- [x] 删除旧的 `detection_pub` 和 `coords_pub`
- [x] 导入 `geometry_msgs.msg.PoseStamped` 和 `json`
- [x] 重写 `publish_detections()` 为 3 个独立发布方法：
  - `_publish_panel_info(panel_normal, panel_centroid)` — 每 N 帧调用
  - `_publish_knobs(knob_detections)` — 每帧调用
  - `_publish_buttons(button_detections)` — 每帧调用

### 2. 目标分拣 — 按 class_id 分离 knob 和 button

- [x] 在 `detection_callback` 中，检测结果按 class_name 分为两组
  ```python
  knobs = []    # class_name == 'knob'
  buttons = []  # class_name == 'indicator_light_on' 或其他
  ```
- [x] 分别传给对应的发布方法

### 3. 稳定 label 生成 — 按空间位置排序

- [x] 新增 `_assign_labels(detections, prefix)` 方法
- [x] 对同类目标按 2D 位置排序（先 y 从上到下分行，同行按 x 从左到右）
- [x] 生成 `knob_0`, `knob_1`, ... 和 `button_0`, `button_1`, ...
- [x] 排序用 bbox 中心点，行分组阈值约为 bbox 高度的 0.5 倍

### 4. config/yolov5s.yaml — 话题配置

- [x] 新增 `ros2_topics` 配置段
  ```yaml
  ros2_topics:
    panel_info: '/panel/info'
    knobs: '/panel/knobs'
    buttons: '/panel/buttons'
  ```

### 5. run.py — 同步改动（可选）

- [x] run.py 不涉及 ROS2，无需改动
- [x] 终端输出中按 knob/button 分组显示（已在 run.py 中集成旋钮角度）

### 6. 测试验证

- [ ] 在 RK3588 上运行 rstest3.py（需连接相机）
- [ ] `ros2 topic echo /panel/info` 确认法向量正确
- [ ] `ros2 topic echo /panel/knobs` 确认 JSON 可解析，角度正确
- [ ] `ros2 topic echo /panel/buttons` 确认按钮位置正确
- [ ] 确认 label 跨帧稳定（同一旋钮编号不跳变）
- [ ] 确认旧话题已移除 ✅

---

## 后续可选优化

- [ ] 如果 JSON String 解析性能不够，引入 colcon 包结构 + 自定义 msg
- [ ] label 从位置排序升级为配置文件映射（"knob_0" → "灯光选择"）
- [ ] 按钮增加 `is_on` 状态判断（颜色分析：亮灯 vs 灭灯）
- [ ] 旋钮增加离散挡位吸附（如 3 档旋钮量化到 0°/120°/240°）
