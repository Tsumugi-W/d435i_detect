'''
rstest3.py
基于 rstest.py，改用操作面板平面拟合法向量（替代按钮平面）
面板点云数量远多于按钮，法向量精度更高
by yzh / extended
'''
from utils.torch_utils import select_device, time_sync
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, set_logging)
from utils.datasets import letterbox
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch

import pyrealsense2 as rs
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

# ── RealSense 初始化 ──────────────────────────────────────────────────────────
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)


# ── 复用 rstest.py 的基础工具函数 ─────────────────────────────────────────────
def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame


def undistort_pixel(u, v, intr, iterations=5):
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy
    coeffs = intr.coeffs
    k1 = coeffs[0] if len(coeffs) > 0 else 0
    k2 = coeffs[1] if len(coeffs) > 1 else 0
    p1 = coeffs[2] if len(coeffs) > 2 else 0
    p2 = coeffs[3] if len(coeffs) > 3 else 0
    k3 = coeffs[4] if len(coeffs) > 4 else 0
    x = (u - cx) / fx
    y = (v - cy) / fy
    for _ in range(iterations):
        r2 = x**2 + y**2
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        dx = 2*p1*x*y + p2*(r2 + 2*x**2)
        dy = p1*(r2 + 2*y**2) + 2*p2*x*y
        x_new = (x - dx) / radial
        y_new = (y - dy) / radial
        if abs(x_new - x) < 1e-6 and abs(y_new - y) < 1e-6:
            break
        x, y = x_new, y_new
    return x * fx + cx, y * fy + cy


def filter_depth(depth_frame, method='bilateral', kernel_size=5):
    depth_image = np.asanyarray(depth_frame.get_data())
    if method == 'bilateral':
        return cv2.bilateralFilter(
            depth_image.astype(np.float32), kernel_size, 75, 75
        ).astype(depth_image.dtype)
    elif method == 'median':
        return cv2.medianBlur(depth_image, kernel_size)
    return depth_image


def get_robust_depth(depth_image, ux, uy, sample_radius=3, depth_scale=0.001):
    depths = []
    for dx in range(-sample_radius, sample_radius + 1):
        for dy in range(-sample_radius, sample_radius + 1):
            x = max(0, min(depth_image.shape[1] - 1, ux + dx))
            y = max(0, min(depth_image.shape[0] - 1, uy + dy))
            d = depth_image[y, x]
            if d > 0:
                depths.append(d * depth_scale)
    return np.median(depths) if depths else 0


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
    """
    RANSAC + SVD 平面拟合，返回 (法向量, 质心) 或 None
    """
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

    策略：
    1. 以所有检测框的联合外扩区域作为面板 ROI
    2. 在 ROI 内，排除所有按钮 bbox 内的像素，只保留面板本身
    3. 用 ROI 边缘像素采样面板颜色，做颜色筛选
    4. 对筛选出的点云做 RANSAC + SVD 平面拟合

    Args:
        color_image: BGR 彩色图
        depth_image: 16位深度图（已滤波）
        depth_intrin: 深度相机内参
        all_bboxes: 所有检测框列表 [(x1,y1,x2,y2), ...]
        panel_color_thresh: 面板颜色筛选阈值
        depth_scale: 深度缩放因子
        sample_stride: 采样步长（跳过像素，加速处理）

    Returns:
        (normal, centroid) 或 None
    """
    if not all_bboxes:
        return None

    h, w = depth_image.shape
    margin = 60  # 向外扩展的像素数，确保包含面板边缘

    # 1. 计算所有 bbox 的联合外扩区域作为面板 ROI
    xs1 = [int(b[0]) for b in all_bboxes]
    ys1 = [int(b[1]) for b in all_bboxes]
    xs2 = [int(b[2]) for b in all_bboxes]
    ys2 = [int(b[3]) for b in all_bboxes]
    roi_x1 = max(0, min(xs1) - margin)
    roi_y1 = max(0, min(ys1) - margin)
    roi_x2 = min(w, max(xs2) + margin)
    roi_y2 = min(h, max(ys2) + margin)

    # 2. 构建按钮遮罩（排除按钮区域）
    button_mask = np.zeros((h, w), dtype=bool)
    for b in all_bboxes:
        bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        button_mask[by1:by2, bx1:bx2] = True

    # 3. 采样面板颜色：取 ROI 四条边缘的像素（避开按钮区域）
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
    panel_color = np.median(edge_pixels, axis=0)  # 用中位数更鲁棒

    # 4. 在 ROI 内筛选面板像素（颜色相近 + 有效深度 + 非按钮区域）
    roi_color = color_image[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
    roi_depth = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_button_mask = button_mask[roi_y1:roi_y2, roi_x1:roi_x2]

    color_diff = np.linalg.norm(roi_color - panel_color, axis=2)
    pixel_mask = (color_diff < panel_color_thresh) & (roi_depth > 0) & (~roi_button_mask)

    # 按步长采样，减少点数（面板点云通常很密，不需要全部）
    ys, xs = np.where(pixel_mask)
    if len(xs) < 50:
        return None
    step_idx = np.arange(0, len(xs), sample_stride)
    ys, xs = ys[step_idx], xs[step_idx]

    # 5. 像素坐标转 3D 点云
    points_3d = []
    for px, py in zip(xs, ys):
        u, v = px + roi_x1, py + roi_y1
        d = roi_depth[py, px] * depth_scale
        pt = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], d)
        points_3d.append(pt)

    points_3d = np.array(points_3d)

    # 6. RANSAC + SVD 平面拟合（阈值稍大，面板可能有轻微弯曲）
    return fit_plane_ransac(points_3d, min_points=50,
                            ransac_iter=100, ransac_thresh=0.008)


# ── YoloV5 检测器（与 rstest.py 相同）────────────────────────────────────────
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


# ── 多帧融合追踪器（与 rstest.py 相同）──────────────────────────────────────
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


# ── ROS2 检测发布节点 ─────────────────────────────────────────────────────────
class DetectionPublisher(Node):
    def __init__(self):
        super().__init__('yolov5_detection_publisher_v3')
        self.detection_pub = self.create_publisher(Detection3DArray, 'detection_3d', 10)
        self.coords_pub = self.create_publisher(String, 'detection_coords', 10)

        self.get_logger().info('YoloV5目标检测(面板法向量版)-程序启动')
        self.model = YoloV5(yolov5_yaml_path='config/yolov5s.yaml')
        self.tracker = MultiFrameTracker(window_size=5, decay_factor=0.8)

        # 缓存面板法向量：面板不动时不需要每帧重算
        self._panel_normal_cache = None
        self._panel_normal_frame_count = 0
        self.panel_normal_update_interval = 10  # 每10帧更新一次面板法向量

        self.timer = self.create_timer(0.033, self.detection_callback)

    def detection_callback(self):
        try:
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()
            if not depth_image.any() or not color_image.any():
                return

            t_start = time.time()
            canvas, class_id_list, xyxy_list, conf_list = self.model.detect(color_image)
            xyxy_list = self.tracker.update(xyxy_list)
            t_end = time.time()

            filtered_depth = filter_depth(aligned_depth_frame, method='bilateral', kernel_size=5)

            # ── 面板法向量（每 N 帧更新一次）────────────────────────────────
            self._panel_normal_frame_count += 1
            if (self._panel_normal_cache is None or
                    self._panel_normal_frame_count % self.panel_normal_update_interval == 0):
                if xyxy_list:
                    result = compute_panel_normal(
                        color_image, filtered_depth, depth_intrin, xyxy_list)
                    if result is not None:
                        self._panel_normal_cache = result

            panel_normal = None
            if self._panel_normal_cache is not None:
                panel_normal, panel_centroid = self._panel_normal_cache
                n_disp = np.round(panel_normal, 3).tolist()
                cv2.putText(canvas, 'panel_n:' + str(n_disp), (10, 80), 0, 0.7,
                            (0, 200, 255), thickness=2, lineType=cv2.LINE_AA)

            # ── 逐目标 3D 坐标 ────────────────────────────────────────────
            camera_xyz_list = []
            for i in range(len(xyxy_list)):
                ux = int((xyxy_list[i][0] + xyxy_list[i][2]) / 2)
                uy = int((xyxy_list[i][1] + xyxy_list[i][3]) / 2)
                ux_u, uy_u = undistort_pixel(ux, uy, intr)
                ux_u, uy_u = int(ux_u), int(uy_u)
                dis = get_robust_depth(filtered_depth, ux_u, uy_u,
                                       sample_radius=3, depth_scale=0.001)
                camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux_u, uy_u), dis)
                camera_xyz = np.round(np.array(camera_xyz), 3).tolist()
                camera_xyz_list.append(camera_xyz)

                cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                cv2.putText(canvas, str(camera_xyz), (ux + 20, uy + 10), 0, 0.7,
                            [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            self.publish_detections(camera_xyz_list, class_id_list, conf_list, panel_normal)

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

    def publish_detections(self, camera_xyz_list, class_id_list, conf_list, panel_normal=None):
        detection_array = Detection3DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera_link'

        # 面板法向量对所有目标共用同一个朝向
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
        coords_msg.data = str({
            'xyz': camera_xyz_list,
            'panel_normal': panel_normal.tolist() if panel_normal is not None else None
        })
        self.coords_pub.publish(coords_msg)
        self.get_logger().info(f'Published {len(camera_xyz_list)} detections')


def main(args=None):
    rclpy.init(args=args)
    try:
        node = DetectionPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
