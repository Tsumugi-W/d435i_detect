'''
rstest3.py
基于 rstest.py，改用操作面板平面拟合法向量（替代按钮平面）
面板点云数量远多于按钮，法向量精度更高
by yzh / extended
适配 RealSense D435i / Orbbec Gemini 336 统一相机后端
'''
from utils.torch_utils import select_device, time_sync
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, set_logging)
from utils.datasets import letterbox
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch

import math
import yaml
import random
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from collections import deque
import json

from depth_utils import (
    deproject_pixel_to_point, deproject_pixels_to_points,
    undistort_pixel, filter_depth, get_robust_depth,
)
import threading
import app_config
from knob_angle import estimate_knob_angle, draw_knob_angle

# 相机实例（由 main() 中初始化）
camera = None


class AsyncCamera:
    """异步取帧：独立线程持续取帧，主线程直接读最新帧"""

    def __init__(self, cam):
        self._cam = cam
        self._frame = (None, None, None, None)
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            result = self._cam.get_aligned_frames()
            if result[0] is not None:
                # 深拷贝帧数据，防止 SDK 复用 buffer 导致主线程读到脏数据
                c_int, d_int, color, depth = result
                with self._lock:
                    self._frame = (c_int, d_int, color.copy(), depth.copy())

    def get_aligned_frames(self):
        with self._lock:
            return self._frame

    def get_depth_scale(self):
        return self._cam.get_depth_scale()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cam.stop()


# ── 硬件温度读取 ─────────────────────────────────────────────────
_THERMAL_ZONES = None

def _read_soc_temp():
    """读取 RK3588 关键温度，返回显示字符串"""
    global _THERMAL_ZONES
    # 首次调用时扫描 thermal zone
    if _THERMAL_ZONES is None:
        _THERMAL_ZONES = {}
        import glob
        for z in sorted(glob.glob('/sys/class/thermal/thermal_zone*')):
            try:
                name = open(f'{z}/type').read().strip()
                if name in ('soc-thermal', 'bigcore0-thermal', 'gpu-thermal', 'npu-thermal'):
                    _THERMAL_ZONES[name] = f'{z}/temp'
            except Exception:
                pass

    parts = []
    for name, path in _THERMAL_ZONES.items():
        try:
            t = int(open(path).read().strip()) / 1000
            short = name.replace('-thermal', '')
            parts.append(f'{short}:{t:.0f}C')
        except Exception:
            pass
    return '  '.join(parts) if parts else ''


# ── 复用基础工具函数 ─────────────────────────────────────────────
def get_aligned_images():
    color_intrin, depth_intrin, color_image, depth_image = camera.get_aligned_frames()
    if color_intrin is None:
        return None, None, None, None, None
    return color_intrin, depth_intrin, color_image, depth_image, depth_image


def normal_to_quaternion(normal):
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
    return [axis[0]*s, axis[1]*s, axis[2]*s, math.cos(angle / 2)]


def fit_plane_ransac(points_3d, min_points=50, ransac_iter=100, ransac_thresh=0.005):
    """RANSAC + SVD 平面拟合，返回 (法向量, 质心) 或 None"""
    if len(points_3d) < min_points:
        return None

    best_normal = None
    best_inliers = 0
    best_inlier_mask = None

    for _ in range(ransac_iter):
        idx = np.random.choice(len(points_3d), 3, replace=False)
        p0, p1, p2 = points_3d[idx]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal = normal / norm_len
        dists = np.abs((points_3d - p0) @ normal)
        inlier_mask = dists < ransac_thresh
        inlier_count = np.sum(inlier_mask)
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_normal = normal
            best_inlier_mask = inlier_mask

    if best_normal is None or best_inliers < min_points:
        return None

    # SVD 精化
    inlier_pts = points_3d[best_inlier_mask]
    centroid = np.mean(inlier_pts, axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid)
    normal = Vt[2]
    if normal[2] > 0:
        normal = -normal
    return normal, centroid


def compute_panel_normal(color_image, depth_image, depth_intrin,
                         all_bboxes, panel_color_thresh=40,
                         depth_scale=0.001, sample_stride=4):
    """
    通过操作面板区域的点云拟合法向量。
    使用向量化反投影替代逐像素 SDK 调用，性能更好。
    """
    if not all_bboxes:
        return None

    h, w = depth_image.shape
    margin = 60

    xs1 = [int(b[0]) for b in all_bboxes]
    ys1 = [int(b[1]) for b in all_bboxes]
    xs2 = [int(b[2]) for b in all_bboxes]
    ys2 = [int(b[3]) for b in all_bboxes]
    roi_x1 = max(0, min(xs1) - margin)
    roi_y1 = max(0, min(ys1) - margin)
    roi_x2 = min(w, max(xs2) + margin)
    roi_y2 = min(h, max(ys2) + margin)

    # 构建按钮遮罩
    button_mask = np.zeros((h, w), dtype=bool)
    for b in all_bboxes:
        bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        button_mask[by1:by2, bx1:bx2] = True

    # 采样面板颜色
    edge_pixels = []
    for x in range(roi_x1, roi_x2, 4):
        for y in [roi_y1, roi_y2 - 1]:
            if not button_mask[y, x]:
                edge_pixels.append(color_image[y, x].astype(np.float32))
    for y in range(roi_y1, roi_y2, 4):
        for x in [roi_x1, roi_x2 - 1]:
            if not button_mask[y, x]:
                edge_pixels.append(color_image[y, x].astype(np.float32))

    if len(edge_pixels) < 10:
        return None
    panel_color = np.median(edge_pixels, axis=0)

    # 在 ROI 内筛选面板像素
    roi_color = color_image[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
    roi_depth = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_button_mask = button_mask[roi_y1:roi_y2, roi_x1:roi_x2]

    color_diff = np.linalg.norm(roi_color - panel_color, axis=2)
    pixel_mask = (color_diff < panel_color_thresh) & (roi_depth > 0) & (~roi_button_mask)

    ys, xs = np.where(pixel_mask)
    if len(xs) < 50:
        return None
    step_idx = np.arange(0, len(xs), sample_stride)
    ys, xs = ys[step_idx], xs[step_idx]

    # 向量化反投影：一次性将所有面板像素转换为 3D 点云
    us = xs + roi_x1
    vs = ys + roi_y1
    depths = roi_depth[ys, xs].astype(np.float64) * depth_scale
    pixels = np.column_stack([us, vs])
    points_3d = deproject_pixels_to_points(depth_intrin, pixels, depths)

    return fit_plane_ransac(points_3d, min_points=50,
                            ransac_iter=100, ransac_thresh=0.008)


# ── YoloV5 检测器 ────────────────────────────────────────────────
class YoloV5:
    def __init__(self, yolov5_yaml_path='config/yolov5s.yaml'):
        with open(yolov5_yaml_path, 'r', encoding='utf-8') as f:
            self.yolov5 = yaml.load(f.read(), Loader=yaml.SafeLoader)
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(self.yolov5['class_num'])]
        self.init_model()

    @torch.no_grad()
    def init_model(self):
        set_logging()
        device = select_device(self.yolov5['device'])
        is_half = device.type != 'cpu'
        model = attempt_load(self.yolov5['weight'], map_location=device)
        check_img_size(self.yolov5['input_size'], s=model.stride.max())
        if is_half:
            model.half()
        if torch.cuda.is_available():
            cudnn.benchmark = True
        img_torch = torch.zeros(
            (1, 3, self.yolov5['input_size'], self.yolov5['input_size']), device=device)
        _ = model(img_torch.half() if is_half else img_torch) if device.type != 'cpu' else None
        self.is_half = is_half
        self.device = device
        self.model = model
        self.img_torch = img_torch

    def preprocessing(self, img):
        img_resize = letterbox(img, new_shape=(
            self.yolov5['input_size'], self.yolov5['input_size']), auto=False)[0]
        img_arr = np.stack([img_resize], 0)
        img_arr = img_arr[:, :, :, ::-1].transpose(0, 3, 1, 2)
        return np.ascontiguousarray(img_arr)

    @torch.no_grad()
    def detect(self, img):
        img_resize = self.preprocessing(img)
        self.img_torch = torch.from_numpy(img_resize).to(self.device)
        self.img_torch = self.img_torch.half() if self.is_half else self.img_torch.float()
        self.img_torch /= 255.0
        if self.img_torch.ndimension() == 3:
            self.img_torch = self.img_torch.unsqueeze(0)
        pred = self.model(self.img_torch, augment=False)[0]
        pred = non_max_suppression(pred, self.yolov5['threshold']['confidence'],
                                   self.yolov5['threshold']['iou'], classes=None, agnostic=False)
        det = pred[0]
        canvas = np.copy(img)
        xyxy_list, conf_list, class_id_list = [], [], []
        if det is not None and len(det):
            det[:, :4] = scale_coords(img_resize.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, class_id in reversed(det):
                class_id = int(class_id)
                xyxy_list.append(xyxy)
                conf_list.append(conf)
                class_id_list.append(class_id)
                label = '%s %.2f' % (self.yolov5['class_name'][class_id], conf)
                self._plot_box(xyxy, canvas, label=label, color=self.colors[class_id])
        return canvas, class_id_list, xyxy_list, conf_list

    def _plot_box(self, x, img, color, label=None, line_thickness=3):
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
        if label:
            tf = max(line_thickness - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# ── 多帧融合追踪器 ──────────────────────────────────────────────
class MultiFrameTracker:
    def __init__(self, window_size=5, decay_factor=0.8):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.history = deque(maxlen=window_size)

    def update(self, detections):
        self.history.append(detections)
        if len(self.history) < 2:
            return detections
        fused = []
        for i, det in enumerate(detections):
            coords, weights = [], []
            for fi, frame_dets in enumerate(self.history):
                if i < len(frame_dets):
                    coords.append(frame_dets[i][:4])
                    weights.append(self.decay_factor ** (len(self.history) - fi - 1))
            if coords:
                w = np.array(weights) / np.sum(weights)
                fused.append(tuple(np.average(coords, axis=0, weights=w)))
            else:
                fused.append(det)
        return fused


# ── 稳定 label 生成 ──────────────────────────────────────────────
def _assign_labels(detections, prefix):
    """
    按空间位置排序并分配稳定标签

    先按 y 从上到下分行（同行 y 差 < bbox 高度 0.5 倍），同行按 x 从左到右。
    生成 prefix_0, prefix_1, ...

    Args:
        detections: [(xyxy, xyz, conf, angle_or_none), ...]
        prefix: 'knob' 或 'button'

    Returns:
        排序后的 detections，每项增加 'label' 字段
    """
    if not detections:
        return []

    # 计算 bbox 中心和典型高度
    items = []
    for det in detections:
        xyxy = det['xyxy']
        cy = (xyxy[1] + xyxy[3]) / 2
        cx = (xyxy[0] + xyxy[2]) / 2
        bh = xyxy[3] - xyxy[1]
        items.append({**det, '_cx': cx, '_cy': cy, '_bh': bh})

    # 按 y 排序
    items.sort(key=lambda d: d['_cy'])

    # 分行：y 差 < 平均高度 * 0.5 认为同行
    avg_bh = np.mean([d['_bh'] for d in items]) if items else 50
    rows = []
    current_row = [items[0]]
    for d in items[1:]:
        if d['_cy'] - current_row[-1]['_cy'] < avg_bh * 0.5:
            current_row.append(d)
        else:
            rows.append(current_row)
            current_row = [d]
    rows.append(current_row)

    # 每行按 x 排序，分配 label
    idx = 0
    result = []
    for row in rows:
        row.sort(key=lambda d: d['_cx'])
        for d in row:
            d['label'] = f'{prefix}_{idx}'
            idx += 1
            result.append(d)

    return result


# ── ROS2 检测发布节点 ─────────────────────────────────────────────
class DetectionPublisher(Node):
    def __init__(self):
        super().__init__('yolov5_detection_publisher_v3')

        # 从配置读取话题名
        topics = app_config.config.get('ros2_topics', {})
        panel_topic = topics.get('panel_info', '/panel/info')
        knobs_topic = topics.get('knobs', '/panel/knobs')
        buttons_topic = topics.get('buttons', '/panel/buttons')

        # 检测结果话题
        self.panel_info_pub = self.create_publisher(PoseStamped, panel_topic, 10)
        self.knobs_pub = self.create_publisher(String, knobs_topic, 10)
        self.buttons_pub = self.create_publisher(String, buttons_topic, 10)

        # 图像话题
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        self._cv_bridge = CvBridge()
        self._camera_info_msg = None  # 延迟构建，等第一帧拿到内参

        self.get_logger().info('YoloV5目标检测(面板法向量版)-程序启动')
        self.get_logger().info(f'检测话题: {panel_topic}, {knobs_topic}, {buttons_topic}')
        self.get_logger().info('图像话题: /camera/color/image_raw, /camera/depth/image_raw')

        # 根据配置选择推理后端
        detector = app_config.create_detector()
        if detector is not None:
            self.model = detector
        else:
            self.model = YoloV5(yolov5_yaml_path='config/yolov5s.yaml')
        self.tracker = MultiFrameTracker(window_size=5, decay_factor=0.8)

        # 获取类别名
        self._class_names = getattr(self.model, 'class_names', None) or \
            app_config.config.get('class_name', [])

        self._panel_normal_cache = None
        self._panel_normal_frame_count = 0
        self.panel_normal_update_interval = 10

        # 旋钮角度估计配置
        angle_cfg = app_config.config.get('knob_angle', {})
        self._angle_enable = angle_cfg.get('enable', False)
        self._angle_binary_thresh = angle_cfg.get('binary_thresh', 180)
        self._angle_circle_mask = angle_cfg.get('circle_mask_ratio', 0.85)
        self._angle_knob_class = angle_cfg.get('knob_class', 'knob')

        self._img_pub_counter = 0
        self._img_pub_interval = 2  # 每 N 帧发布一次图像（降频，避免卡顿）

        self._display_frame = None  # 供主线程显示的画面

        self.timer = self.create_timer(0.033, self.detection_callback)

    def _build_camera_info(self, intrin):
        """从内参构建 CameraInfo 消息（只构建一次）"""
        msg = CameraInfo()
        msg.header.frame_id = 'camera_link'
        msg.width = intrin.width
        msg.height = intrin.height
        msg.distortion_model = 'plumb_bob'
        msg.d = [float(c) for c in intrin.coeffs]
        msg.k = [intrin.fx, 0.0, intrin.cx,
                  0.0, intrin.fy, intrin.cy,
                  0.0, 0.0, 1.0]
        msg.p = [intrin.fx, 0.0, intrin.cx, 0.0,
                  0.0, intrin.fy, intrin.cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        return msg

    def _publish_images(self, color_image, depth_image, now, intr):
        """发布彩色图、深度图和相机内参"""
        stamp = now.to_msg()

        # 彩色图
        color_msg = self._cv_bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        color_msg.header.stamp = stamp
        color_msg.header.frame_id = 'camera_link'
        self.color_pub.publish(color_msg)

        # 深度图 (16UC1, 单位 mm)
        depth_msg = self._cv_bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = 'camera_link'
        self.depth_pub.publish(depth_msg)

        # 相机内参（延迟构建）
        if self._camera_info_msg is None:
            self._camera_info_msg = self._build_camera_info(intr)
        self._camera_info_msg.header.stamp = stamp
        self.camera_info_pub.publish(self._camera_info_msg)

    def detection_callback(self):
        try:
            intr, depth_intrin, color_image, depth_image, _ = get_aligned_images()
            if intr is None or not depth_image.any() or not color_image.any():
                return

            # 发布图像话题（降频，避免序列化开销拖慢主循环）
            now = self.get_clock().now()
            self._img_pub_counter += 1
            if self._img_pub_counter % self._img_pub_interval == 0:
                self._publish_images(color_image, depth_image, now, intr)

            t_start = time.time()
            canvas, class_id_list, xyxy_list, conf_list = self.model.detect(color_image)
            xyxy_list = self.tracker.update(xyxy_list)
            t_end = time.time()

            filtered_depth = filter_depth(depth_image, method='bilateral', kernel_size=5)
            depth_scale = camera.get_depth_scale()

            # ── 面板法向量（每 N 帧更新一次）────────────────────────
            self._panel_normal_frame_count += 1
            if (self._panel_normal_cache is None or
                    self._panel_normal_frame_count % self.panel_normal_update_interval == 0):
                if xyxy_list:
                    result = compute_panel_normal(
                        color_image, filtered_depth, depth_intrin, xyxy_list,
                        depth_scale=depth_scale)
                    if result is not None:
                        self._panel_normal_cache = result

            panel_normal, panel_centroid = None, None
            if self._panel_normal_cache is not None:
                panel_normal, panel_centroid = self._panel_normal_cache
                n_disp = np.round(panel_normal, 3).tolist()
                cv2.putText(canvas, 'panel_n:' + str(n_disp), (10, 80), 0, 0.7,
                            (0, 200, 255), thickness=2, lineType=cv2.LINE_AA)

            # ── 逐目标 3D 坐标 + 分类 ────────────────────────────
            knobs_raw, buttons_raw = [], []

            for i in range(len(xyxy_list)):
                ux = int((xyxy_list[i][0] + xyxy_list[i][2]) / 2)
                uy = int((xyxy_list[i][1] + xyxy_list[i][3]) / 2)
                ux_u, uy_u = undistort_pixel(ux, uy, intr)
                ux_u, uy_u = int(ux_u), int(uy_u)
                dis = get_robust_depth(filtered_depth, ux_u, uy_u,
                                       sample_radius=3, depth_scale=depth_scale)
                camera_xyz = deproject_pixel_to_point(depth_intrin, (ux_u, uy_u), dis)
                camera_xyz = np.round(np.array(camera_xyz), 3).tolist()

                cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                cv2.putText(canvas, str(camera_xyz), (ux + 20, uy + 10), 0, 0.7,
                            [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                # 获取类别名
                cls_name = self._class_names[class_id_list[i]] \
                    if i < len(class_id_list) and class_id_list[i] < len(self._class_names) \
                    else ''
                conf = float(conf_list[i]) if i < len(conf_list) else 0.0

                det_item = {
                    'xyxy': list(map(float, xyxy_list[i])),
                    'xyz': camera_xyz,
                    'confidence': conf,
                }

                # 旋钮：估计角度
                if cls_name == self._angle_knob_class:
                    angle = None
                    if self._angle_enable:
                        x1, y1 = int(xyxy_list[i][0]), int(xyxy_list[i][1])
                        x2, y2 = int(xyxy_list[i][2]), int(xyxy_list[i][3])
                        roi = color_image[y1:y2, x1:x2]
                        angle = estimate_knob_angle(
                            roi,
                            binary_thresh=self._angle_binary_thresh,
                            circle_mask_ratio=self._angle_circle_mask,
                        )
                        if angle is not None:
                            draw_knob_angle(canvas, xyxy_list[i], angle)
                    det_item['angle'] = angle
                    knobs_raw.append(det_item)
                else:
                    buttons_raw.append(det_item)

            # ── 分配稳定 label ────────────────────────────────────
            knobs = _assign_labels(knobs_raw, 'knob')
            buttons = _assign_labels(buttons_raw, 'button')

            # ── 发布 ─────────────────────────────────────────────
            now = self.get_clock().now()
            stamp_sec = now.nanoseconds / 1e9

            self._publish_panel_info(panel_normal, panel_centroid, now)
            self._publish_knobs(knobs, stamp_sec)
            self._publish_buttons(buttons, stamp_sec)

            n_total = len(knobs) + len(buttons)
            fps = int(1.0 / max(t_end - t_start, 1e-6))
            cv2.putText(canvas, 'FPS: {}'.format(fps), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            temp_str = _read_soc_temp()
            if temp_str:
                cv2.putText(canvas, temp_str, (50, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # 存入共享变量，由主线程显示（避免 GUI 卡死）
            self._display_frame = canvas

        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def _publish_panel_info(self, panel_normal, panel_centroid, now):
        """发布面板位姿 PoseStamped"""
        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'camera_link'

        if panel_centroid is not None:
            msg.pose.position.x = float(panel_centroid[0])
            msg.pose.position.y = float(panel_centroid[1])
            msg.pose.position.z = float(panel_centroid[2])

        if panel_normal is not None:
            quat = normal_to_quaternion(panel_normal)
            msg.pose.orientation.x = quat[0]
            msg.pose.orientation.y = quat[1]
            msg.pose.orientation.z = quat[2]
            msg.pose.orientation.w = quat[3]

        self.panel_info_pub.publish(msg)

    def _publish_knobs(self, knobs, stamp_sec):
        """发布旋钮信息 JSON String"""
        data = {
            'stamp': stamp_sec,
            'knobs': [
                {
                    'label': k['label'],
                    'position': {'x': k['xyz'][0], 'y': k['xyz'][1], 'z': k['xyz'][2]},
                    'angle': k.get('angle'),
                    'confidence': k['confidence'],
                }
                for k in knobs
            ]
        }
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.knobs_pub.publish(msg)

    def _publish_buttons(self, buttons, stamp_sec):
        """发布按钮信息 JSON String"""
        data = {
            'stamp': stamp_sec,
            'buttons': [
                {
                    'label': b['label'],
                    'position': {'x': b['xyz'][0], 'y': b['xyz'][1], 'z': b['xyz'][2]},
                    'confidence': b['confidence'],
                }
                for b in buttons
            ]
        }
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.buttons_pub.publish(msg)


def main(args=None):
    global camera
    app_config.load_config()
    raw_cam = app_config.init_camera()
    camera = AsyncCamera(raw_cam)

    rclpy.init(args=args)
    node = DetectionPublisher()

    # ROS2 spin 放后台线程，主线程专跑 OpenCV GUI
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        cv2.namedWindow('detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        while rclpy.ok():
            if node._display_frame is not None:
                cv2.imshow('detection', node._display_frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
