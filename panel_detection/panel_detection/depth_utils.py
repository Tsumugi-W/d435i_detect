"""
公共深度处理工具函数

从 rstest.py / rstest3.py 提取的与相机 SDK 无关的纯数学/图像处理函数。
适用于 RealSense D435i / Orbbec Gemini 336 等任意深度相机。
"""
import numpy as np
import cv2

from .camera.base import CameraIntrinsics


def deproject_pixel_to_point(intrin: CameraIntrinsics,
                             pixel: tuple, depth: float) -> list:
    """
    将像素坐标 + 深度 反投影为 3D 相机坐标（pinhole 模型）

    等价于 rs2_deproject_pixel_to_point()，但不依赖任何 SDK。

    Args:
        intrin: 相机内参（通常用深度相机内参）
        pixel: (u, v) 像素坐标
        depth: 深度值（米）

    Returns:
        [x, y, z] 相机坐标系下的 3D 点（米）
    """
    x = (pixel[0] - intrin.cx) / intrin.fx
    y = (pixel[1] - intrin.cy) / intrin.fy
    return [depth * x, depth * y, depth]


def deproject_pixels_to_points(intrin: CameraIntrinsics,
                               pixels: np.ndarray,
                               depths: np.ndarray) -> np.ndarray:
    """
    向量化批量反投影：将 N 个像素坐标 + 深度转为 3D 点云

    Args:
        intrin: 相机内参
        pixels: (N, 2) 数组，每行 [u, v]
        depths: (N,) 数组，深度值（米）

    Returns:
        (N, 3) 数组，每行 [x, y, z]
    """
    us = pixels[:, 0].astype(np.float64)
    vs = pixels[:, 1].astype(np.float64)
    x = (us - intrin.cx) / intrin.fx * depths
    y = (vs - intrin.cy) / intrin.fy * depths
    return np.column_stack([x, y, depths])


def undistort_pixel(u, v, intrin: CameraIntrinsics, iterations=5):
    """
    将畸变的像素坐标转换为理想坐标（去畸变）

    使用 Brown-Conrady 畸变模型的迭代反演（牛顿法）

    Args:
        u, v: 实际像素坐标
        intrin: 统一内参对象
        iterations: 迭代次数（通常 3-5 次收敛）

    Returns:
        (u_ideal, v_ideal): 去畸变后的理想像素坐标
    """
    fx, fy = intrin.fx, intrin.fy
    cx, cy = intrin.cx, intrin.cy

    coeffs = intrin.coeffs
    k1 = coeffs[0] if len(coeffs) > 0 else 0
    k2 = coeffs[1] if len(coeffs) > 1 else 0
    p1 = coeffs[2] if len(coeffs) > 2 else 0
    p2 = coeffs[3] if len(coeffs) > 3 else 0
    k3 = coeffs[4] if len(coeffs) > 4 else 0

    # 转换为归一化坐标
    x = (u - cx) / fx
    y = (v - cy) / fy

    # 迭代求解理想坐标
    for _ in range(iterations):
        r2 = x**2 + y**2
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        dy = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

        x_new = (x - dx) / radial
        y_new = (y - dy) / radial

        if abs(x_new - x) < 1e-6 and abs(y_new - y) < 1e-6:
            break
        x, y = x_new, y_new

    return x * fx + cx, y * fy + cy


def filter_depth(depth_image: np.ndarray, method='bilateral', kernel_size=5):
    """
    对深度图进行滤波降噪

    Args:
        depth_image: 深度图 numpy 数组 (H, W) uint16
        method: 滤波方法 'bilateral' | 'median' | 'gaussian'
        kernel_size: 核大小（必须为奇数）

    Returns:
        滤波后的深度图 numpy 数组
    """
    if method == 'bilateral':
        return cv2.bilateralFilter(
            depth_image.astype(np.float32), kernel_size, 75, 75
        ).astype(depth_image.dtype)
    elif method == 'median':
        return cv2.medianBlur(depth_image, kernel_size)
    elif method == 'gaussian':
        return cv2.GaussianBlur(depth_image, (kernel_size, kernel_size), 0)
    return depth_image


def get_robust_depth(depth_image, ux, uy, sample_radius=3, depth_scale=0.001):
    """
    从中心点周围采样多个深度值，取中位数

    Args:
        depth_image: 深度图 numpy 数组
        ux, uy: 中心像素坐标
        sample_radius: 采样半径（像素）
        depth_scale: 深度缩放因子

    Returns:
        鲁棒的深度值（米）
    """
    depths = []
    for dx in range(-sample_radius, sample_radius + 1):
        for dy in range(-sample_radius, sample_radius + 1):
            x = max(0, min(depth_image.shape[1] - 1, ux + dx))
            y = max(0, min(depth_image.shape[0] - 1, uy + dy))
            d = depth_image[y, x]
            if d > 0:
                depths.append(d * depth_scale)
    return np.median(depths) if depths else 0


def fit_plane_ransac(points_3d, min_points=50, ransac_iter=100, ransac_thresh=0.005):
    """
    RANSAC + SVD 平面拟合

    Args:
        points_3d: (N, 3) 点云数组
        min_points: 最少内点数
        ransac_iter: RANSAC 迭代次数
        ransac_thresh: 内点距离阈值（米）

    Returns:
        (normal, centroid) 或 None
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
    通过操作面板区域的点云拟合法向量

    策略：
    1. 以所有检测框的联合外扩区域作为面板 ROI
    2. 排除按钮 bbox 内的像素，只保留面板本身
    3. 用 ROI 边缘像素采样面板颜色，做颜色筛选
    4. 向量化反投影为 3D 点云，RANSAC + SVD 拟合平面

    Args:
        color_image: BGR 彩色图
        depth_image: 深度图 (H, W) uint16（已滤波）
        depth_intrin: 深度相机内参 CameraIntrinsics
        all_bboxes: 检测框列表 [(x1,y1,x2,y2), ...]
        panel_color_thresh: 面板颜色筛选阈值
        depth_scale: 深度缩放因子
        sample_stride: 采样步长

    Returns:
        (normal, centroid) 或 None
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

    # 向量化反投影
    us = xs + roi_x1
    vs = ys + roi_y1
    depths = roi_depth[ys, xs].astype(np.float64) * depth_scale
    pixels = np.column_stack([us, vs])
    points_3d = deproject_pixels_to_points(depth_intrin, pixels, depths)

    return fit_plane_ransac(points_3d, min_points=50,
                            ransac_iter=100, ransac_thresh=0.008)
