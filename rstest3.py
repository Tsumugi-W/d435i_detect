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
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from collections import deque

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
                with self._lock:
                    self._frame = result

    def get_aligned_frames(self):
        with self._lock:
            return self._frame

    def get_depth_scale(self):
        return self._cam.get_depth_scale()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cam.stop()


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


# ── ROS2 检测发布节点 ─────────────────────────────────────────────
class DetectionPublisher(Node):
    def __init__(self):
        super().__init__('yolov5_detection_publisher_v3')
        self.detection_pub = self.create_publisher(Detection3DArray, 'detection_3d', 10)
        self.coords_pub = self.create_publisher(String, 'detection_coords', 10)

        self.get_logger().info('YoloV5目标检测(面板法向量版)-程序启动')
        # 根据配置选择推理后端
        rknn_detector = app_config.create_detector()
        if rknn_detector is not None:
            self.model = rknn_detector
        else:
            self.model = YoloV5(yolov5_yaml_path='config/yolov5s.yaml')
        self.tracker = MultiFrameTracker(window_size=5, decay_factor=0.8)

        self._panel_normal_cache = None
        self._panel_normal_frame_count = 0
        self.panel_normal_update_interval = 10

        # 旋钮角度估计配置
        angle_cfg = app_config.config.get('knob_angle', {})
        self._angle_enable = angle_cfg.get('enable', False)
        self._angle_binary_thresh = angle_cfg.get('binary_thresh', 180)
        self._angle_circle_mask = angle_cfg.get('circle_mask_ratio', 0.85)
        self._angle_knob_class = angle_cfg.get('knob_class', 'knob')

        self.timer = self.create_timer(0.033, self.detection_callback)

    def detection_callback(self):
        try:
            intr, depth_intrin, color_image, depth_image, _ = get_aligned_images()
            if intr is None or not depth_image.any() or not color_image.any():
                return

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

            panel_normal = None
            if self._panel_normal_cache is not None:
                panel_normal, panel_centroid = self._panel_normal_cache
                n_disp = np.round(panel_normal, 3).tolist()
                cv2.putText(canvas, 'panel_n:' + str(n_disp), (10, 80), 0, 0.7,
                            (0, 200, 255), thickness=2, lineType=cv2.LINE_AA)

            # ── 逐目标 3D 坐标 ────────────────────────────────────
            camera_xyz_list = []
            for i in range(len(xyxy_list)):
                ux = int((xyxy_list[i][0] + xyxy_list[i][2]) / 2)
                uy = int((xyxy_list[i][1] + xyxy_list[i][3]) / 2)
                ux_u, uy_u = undistort_pixel(ux, uy, intr)
                ux_u, uy_u = int(ux_u), int(uy_u)
                dis = get_robust_depth(filtered_depth, ux_u, uy_u,
                                       sample_radius=3, depth_scale=depth_scale)
                camera_xyz = deproject_pixel_to_point(depth_intrin, (ux_u, uy_u), dis)
                camera_xyz = np.round(np.array(camera_xyz), 3).tolist()
                camera_xyz_list.append(camera_xyz)

                cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                cv2.putText(canvas, str(camera_xyz), (ux + 20, uy + 10), 0, 0.7,
                            [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            # ── 旋钮角度估计 ─────────────────────────────────────
            angle_list = [None] * len(xyxy_list)
            if self._angle_enable:
                class_names = getattr(self.model, 'class_names', None) or \
                    app_config.config.get('class_name', [])
                for i, xyxy in enumerate(xyxy_list):
                    cls_name = class_names[class_id_list[i]] \
                        if class_id_list[i] < len(class_names) else ''
                    if cls_name != self._angle_knob_class:
                        continue
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    x2, y2 = int(xyxy[2]), int(xyxy[3])
                    roi = color_image[y1:y2, x1:x2]
                    angle = estimate_knob_angle(
                        roi,
                        binary_thresh=self._angle_binary_thresh,
                        circle_mask_ratio=self._angle_circle_mask,
                    )
                    if angle is not None:
                        angle_list[i] = angle
                        draw_knob_angle(canvas, xyxy, angle)

            self.publish_detections(camera_xyz_list, class_id_list, conf_list,
                                    panel_normal, angle_list)

            fps = int(1.0 / max(t_end - t_start, 1e-6))
            cv2.putText(canvas, 'FPS: {}'.format(fps), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.namedWindow('detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow('detection', canvas)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                cv2.destroyAllWindows()
                raise KeyboardInterrupt

        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def publish_detections(self, camera_xyz_list, class_id_list, conf_list,
                           panel_normal=None, angle_list=None):
        detection_array = Detection3DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera_link'

        quat = normal_to_quaternion(panel_normal) if panel_normal is not None else None

        for i, xyz in enumerate(camera_xyz_list):
            detection = Detection3D()
            detection.bbox.center.position.x = float(xyz[0])
            detection.bbox.center.position.y = float(xyz[1])
            detection.bbox.center.position.z = float(xyz[2])
            if quat is not None:
                detection.bbox.center.orientation.x = quat[0]
                detection.bbox.center.orientation.y = quat[1]
                detection.bbox.center.orientation.z = quat[2]
                detection.bbox.center.orientation.w = quat[3]
            if i < len(class_id_list):
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(class_id_list[i])
                hyp.hypothesis.score = float(conf_list[i])
                detection.results.append(hyp)
            detection_array.detections.append(detection)

        self.detection_pub.publish(detection_array)

        coords_msg = String()
        angles = [angle_list[i] if angle_list else None
                  for i in range(len(camera_xyz_list))]
        coords_msg.data = str({
            'xyz': camera_xyz_list,
            'panel_normal': panel_normal.tolist() if panel_normal is not None else None,
            'knob_angles': angles,
        })
        self.coords_pub.publish(coords_msg)
        self.get_logger().info(f'Published {len(camera_xyz_list)} detections')


def main(args=None):
    global camera
    app_config.load_config()
    raw_cam = app_config.init_camera()
    camera = AsyncCamera(raw_cam)

    rclpy.init(args=args)
    try:
        node = DetectionPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
