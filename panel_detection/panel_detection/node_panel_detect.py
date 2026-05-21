"""
ROS2 面板位姿检测节点

发布话题:
  /panel/info     (geometry_msgs/PoseStamped) — 面板位姿（中心点 + 法向量四元数）
  /panel/buttons  (std_msgs/String, JSON)     — 按钮位置
  /panel/knobs    (std_msgs/String, JSON)     — 旋钮位置 + 角度
  /panel/bolts    (std_msgs/String, JSON)     — 螺栓位置
  /panel/nuts     (std_msgs/String, JSON)     — 螺母位置
  /panel/valves   (std_msgs/String, JSON)     — 阀门位置
  /panel/pumps    (std_msgs/String, JSON)     — 泵位置
"""
import os
import math
import json
import time
import threading

import yaml
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

# 包内导入
from .camera import create_backend
from .camera.base import CameraIntrinsics
from .depth_utils import (
    deproject_pixel_to_point, undistort_pixel,
    filter_depth, get_robust_depth, compute_panel_normal,
)
from .knob_angle import estimate_knob_angle


class AsyncCamera:
    def __init__(self, cam):
        self._cam = cam
        self._frame = (None, None, None, None)
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                result = self._cam.get_aligned_frames()
                if result[0] is not None:
                    with self._lock:
                        self._frame = result
            except Exception:
                time.sleep(0.1)

    def get_aligned_frames(self):
        with self._lock:
            return self._frame

    def get_depth_scale(self):
        return self._cam.get_depth_scale()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cam.stop()


def _normal_to_quaternion(normal):
    """将法向量转换为四元数 (x, y, z, w)"""
    n = np.array(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    ref = np.array([0.0, 0.0, -1.0])
    cross = np.cross(ref, n)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-10:
        return [0.0, 0.0, 0.0, 1.0] if np.dot(ref, n) > 0 else [1.0, 0.0, 0.0, 0.0]
    axis = cross / cross_norm
    angle = math.acos(np.clip(np.dot(ref, n), -1.0, 1.0))
    s = math.sin(angle / 2)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)]


def _assign_labels(detections, prefix):
    """
    按空间位置排序并生成稳定 label

    排序规则：先按 y 从上到下分行，同行按 x 从左到右
    """
    if not detections:
        return detections

    avg_h = np.mean([d['bbox_h'] for d in detections])
    row_thresh = avg_h * 0.5

    sorted_dets = sorted(detections, key=lambda d: d['cy'])

    rows = []
    current_row = [sorted_dets[0]]
    for d in sorted_dets[1:]:
        if abs(d['cy'] - current_row[0]['cy']) < row_thresh:
            current_row.append(d)
        else:
            rows.append(current_row)
            current_row = [d]
    rows.append(current_row)

    idx = 0
    for row in rows:
        row.sort(key=lambda d: d['cx'])
        for d in row:
            d['label'] = f'{prefix}_{idx}'
            idx += 1

    return detections


class PanelDetectionNode(Node):
    def __init__(self):
        super().__init__('panel_detection_node')

        # 声明参数
        self.declare_parameter('config_path', '')
        config_path = self.get_parameter('config_path').get_parameter_value().string_value

        if not config_path:
            # 从 ament share 目录查找配置文件
            try:
                from ament_index_python.packages import get_package_share_directory
                config_path = os.path.join(
                    get_package_share_directory('panel_detection'), 'config', 'panel_detection.yaml')
            except Exception:
                # fallback: 相对于源码目录
                config_path = os.path.join(
                    os.path.dirname(__file__), '..', 'config', 'panel_detection.yaml')

        self.get_logger().info(f'加载配置: {config_path}')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)

        # 话题配置
        topics = self.cfg.get('ros2_topics', {})
        self.panel_info_pub = self.create_publisher(
            PoseStamped, topics.get('panel_info', '/panel/info'), 10)

        # 为每个类别创建独立 publisher
        self._topic_pub_map = {}
        for topic_key, topic_name in topics.items():
            if topic_key == 'panel_info':
                continue
            pub = self.create_publisher(String, topic_name, 10)
            self._topic_pub_map[topic_key] = pub

        # 类别到话题的映射
        self._class_topic_map = self.cfg.get('class_topic_mapping', {
            'button': 'buttons',
            'knob': 'knobs',
            'bolt': 'bolts',
            'nut': 'nuts',
            'valve': 'valves',
            'pump': 'pumps',
        })

        # 初始化相机（支持重试，等待设备连接）
        backend_type = self.cfg.get('camera_backend', 'orbbec')
        cam_cfg = self.cfg.get('camera', {})
        self._camera = None
        self._depth_scale = 0.001
        self._camera_ready = False

        max_retries = 10
        retry_interval = 3.0
        for attempt in range(1, max_retries + 1):
            try:
                raw_cam = create_backend(backend_type)
                raw_cam.initialize(
                    cam_cfg.get('color_width', 640),
                    cam_cfg.get('color_height', 480),
                    cam_cfg.get('depth_width', 640),
                    cam_cfg.get('depth_height', 480),
                    cam_cfg.get('fps', 30),
                )
                self._camera = AsyncCamera(raw_cam)
                self._depth_scale = self._camera.get_depth_scale()
                self._camera_ready = True
                self.get_logger().info(
                    f'相机已启动: {backend_type}, depth_scale={self._depth_scale}')
                break
            except Exception as e:
                self.get_logger().warn(
                    f'相机初始化失败 (尝试 {attempt}/{max_retries}): {e}')
                if attempt < max_retries:
                    self.get_logger().info(
                        f'{retry_interval}秒后重试...')
                    time.sleep(retry_interval)
                else:
                    self.get_logger().error(
                        f'相机初始化失败，已重试 {max_retries} 次。节点将等待相机连接...')

        # 初始化检测器
        self._detector = self._create_detector(config_path)
        self._class_names = getattr(self._detector, 'class_names', None) or \
            self.cfg.get('class_name', [])

        # 旋钮角度估计
        angle_cfg = self.cfg.get('knob_angle', {})
        self._angle_enable = angle_cfg.get('enable', False)
        self._angle_binary_thresh = angle_cfg.get('binary_thresh', 180)
        self._angle_circle_mask = angle_cfg.get('circle_mask_ratio', 0.85)
        self._angle_knob_class = angle_cfg.get('knob_class', 'knob')

        # 面板法向量缓存
        self._panel_normal_cache = None
        self._frame_count = 0
        self._normal_interval = self.cfg.get('panel_normal_interval', 10)

        # 重连相关
        self._reconnect_interval = 5.0  # 秒
        self._last_reconnect_time = 0.0

        # 定时回调 (~30Hz)
        self._timer = self.create_timer(0.033, self._detection_callback)
        if self._camera_ready:
            self.get_logger().info('面板检测节点已启动')
        else:
            self.get_logger().info('面板检测节点已启动（等待相机连接）')

    def _create_detector(self, config_path):
        backend = self.cfg.get('inference_backend', 'onnx')

        if backend == 'rknn':
            from .detector_rknn import YoloV5RKNN
            rknn_model = self.cfg.get('rknn_model', 'weights/best.rknn')
            self.get_logger().info(f'推理后端: RKNN NPU ({rknn_model})')
            return YoloV5RKNN(rknn_model_path=rknn_model, config_path=config_path)

        if backend == 'onnx':
            from .detector_onnx import YoloV5ORT
            onnx_path = self.cfg.get('onnx_model', 'weights/best.onnx')
            threads = self.cfg.get('onnx_threads', 4)
            self.get_logger().info(f'推理后端: ONNX Runtime ({onnx_path})')
            det = YoloV5ORT(onnx_path=onnx_path, config_path=config_path, threads=threads)
            pt_path = self.cfg.get('weight', 'weights/best.pt')
            self._try_load_names(det, pt_path)
            return det

        self.get_logger().info('推理后端: PyTorch')
        return None

    def _try_load_names(self, detector, pt_path):
        try:
            import torch
            ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
            model = ckpt.get('ema') or ckpt.get('model')
            if hasattr(model, 'names') and model.names:
                names = model.names
                if isinstance(names, dict):
                    detector.class_names = [names[i] for i in sorted(names.keys())]
                else:
                    detector.class_names = list(names)
                detector.class_num = len(detector.class_names)
                import random
                detector.colors = [[random.randint(0, 255) for _ in range(3)]
                                   for _ in range(detector.class_num)]
                self.get_logger().info(f'模型类别: {detector.class_names}')
        except Exception as e:
            self.get_logger().warn(f'无法从 {pt_path} 加载类别名: {e}')

    def _try_reconnect_camera(self):
        """定时尝试重新连接相机"""
        now = time.time()
        if now - self._last_reconnect_time < self._reconnect_interval:
            return
        self._last_reconnect_time = now

        backend_type = self.cfg.get('camera_backend', 'orbbec')
        cam_cfg = self.cfg.get('camera', {})
        try:
            raw_cam = create_backend(backend_type)
            raw_cam.initialize(
                cam_cfg.get('color_width', 640),
                cam_cfg.get('color_height', 480),
                cam_cfg.get('depth_width', 640),
                cam_cfg.get('depth_height', 480),
                cam_cfg.get('fps', 30),
            )
            self._camera = AsyncCamera(raw_cam)
            self._depth_scale = self._camera.get_depth_scale()
            self._camera_ready = True
            self.get_logger().info(
                f'相机已连接: {backend_type}, depth_scale={self._depth_scale}')
        except Exception:
            pass  # 静默等待下次重试

    def _detection_callback(self):
        if not self._camera_ready:
            # 尝试重新连接相机
            self._try_reconnect_camera()
            return

        color_intrin, depth_intrin, color_image, depth_image = \
            self._camera.get_aligned_frames()
        if color_intrin is None:
            return
        if not color_image.any() or not depth_image.any():
            return

        canvas, class_id_list, xyxy_list, conf_list = self._detector.detect(color_image)
        if not xyxy_list:
            return

        filtered_depth = filter_depth(depth_image, method='bilateral', kernel_size=5)
        self._depth_scale = self._camera.get_depth_scale()

        # 面板法向量
        self._frame_count += 1
        if (self._panel_normal_cache is None or
                self._frame_count % self._normal_interval == 0):
            result = compute_panel_normal(
                color_image, filtered_depth, depth_intrin, xyxy_list,
                depth_scale=self._depth_scale)
            if result is not None:
                self._panel_normal_cache = result

        if self._panel_normal_cache is not None:
            self._publish_panel_info(*self._panel_normal_cache)

        # 按类别分组
        grouped = {}
        for i, xyxy in enumerate(xyxy_list):
            cls_id = class_id_list[i]
            cls_name = self._class_names[cls_id] if cls_id < len(self._class_names) else 'unknown'
            topic_key = self._class_topic_map.get(cls_name)
            if topic_key is None:
                continue

            ux = int((xyxy[0] + xyxy[2]) / 2)
            uy = int((xyxy[1] + xyxy[3]) / 2)
            ux_u, uy_u = undistort_pixel(ux, uy, color_intrin)
            ux_u, uy_u = int(ux_u), int(uy_u)
            dis = get_robust_depth(filtered_depth, ux_u, uy_u,
                                   sample_radius=3, depth_scale=self._depth_scale)
            xyz = deproject_pixel_to_point(depth_intrin, (ux_u, uy_u), dis)

            det = {
                'class': cls_name,
                'position': {
                    'x': round(xyz[0], 4),
                    'y': round(xyz[1], 4),
                    'z': round(xyz[2], 4),
                },
                'confidence': round(float(conf_list[i]), 3),
                'cx': ux,
                'cy': uy,
                'bbox_h': int(xyxy[3] - xyxy[1]),
            }

            if cls_name == self._angle_knob_class and self._angle_enable:
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                x2, y2 = int(xyxy[2]), int(xyxy[3])
                roi = color_image[y1:y2, x1:x2]
                angle = estimate_knob_angle(
                    roi,
                    binary_thresh=self._angle_binary_thresh,
                    circle_mask_ratio=self._angle_circle_mask,
                )
                if angle is not None:
                    det['angle'] = round(angle, 1)

            grouped.setdefault(topic_key, []).append(det)

        stamp = time.time()
        for topic_key, dets in grouped.items():
            prefix = topic_key.rstrip('s')
            _assign_labels(dets, prefix)
            self._publish_detections(topic_key, dets, stamp)

    def _publish_panel_info(self, normal, centroid):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.pose.position.x = float(centroid[0])
        msg.pose.position.y = float(centroid[1])
        msg.pose.position.z = float(centroid[2])
        quat = _normal_to_quaternion(normal)
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        self.panel_info_pub.publish(msg)

    def _publish_detections(self, topic_key, detections, stamp):
        pub = self._topic_pub_map.get(topic_key)
        if pub is None:
            return

        items = []
        for d in detections:
            item = {
                'label': d.get('label', ''),
                'class': d['class'],
                'position': d['position'],
                'confidence': d['confidence'],
            }
            if 'angle' in d:
                item['angle'] = d['angle']
            items.append(item)

        payload = {
            'stamp': round(stamp, 3),
            topic_key: items,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PanelDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if node._camera is not None:
                node._camera.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
