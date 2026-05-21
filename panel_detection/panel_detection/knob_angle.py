"""
旋钮角度估计模块

通过传统 CV 方法检测旋钮上的白色指针条方向，估计旋转角度。
仅依赖 OpenCV + numpy，无需额外模型。

用法:
    from knob_angle import estimate_knob_angle
    angle = estimate_knob_angle(knob_roi)  # 输入 BGR 裁剪图，返回角度 (°) 或 None
"""
import math
import cv2
import numpy as np


def estimate_knob_angle(
    color_roi: np.ndarray,
    binary_thresh: int = 180,
    min_pointer_area_ratio: float = 0.01,
    max_pointer_area_ratio: float = 0.25,
    circle_mask_ratio: float = 0.85,
    debug: bool = False,
) -> float | None:
    """
    估计旋钮指针角度

    自动判断旋钮类型：
      - 白色指针旋钮（黑色旋钮上有白色标记）：灰度二值化提取
      - 彩色把手旋钮（红色/棕色旋转把手）：HSV 颜色隔离提取

    Args:
        color_roi: BGR 格式的旋钮裁剪图 (来自 YOLO bbox)
        binary_thresh: 二值化阈值，用于提取白色指针（高亮区域）
        min_pointer_area_ratio: 指针最小面积占圆形区域的比例
        max_pointer_area_ratio: 指针最大面积占圆形区域的比例
        circle_mask_ratio: 圆形 mask 半径相对于 ROI 短边半径的比例
                           用于去除旋钮边框（银色金属圈）干扰
        debug: 是否返回调试中间结果

    Returns:
        角度值 (°)，以 12 点钟方向为 0°，顺时针增加，范围 [0, 360)
        如果无法检测到有效指针则返回 None
        当 debug=True 时返回 (angle, debug_dict)
    """
    if color_roi is None or color_roi.size == 0:
        return (None, {}) if debug else None

    h, w = color_roi.shape[:2]
    if h < 10 or w < 10:
        return (None, {}) if debug else None

    cx, cy = w // 2, h // 2
    radius = int(min(cx, cy) * circle_mask_ratio)

    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), radius, 255, -1)
    circle_area = math.pi * radius * radius

    # 先尝试白色指针方法
    angle, pointer_contour, binary = _try_white_pointer(
        color_roi, circle_mask, radius, circle_area,
        binary_thresh, min_pointer_area_ratio, max_pointer_area_ratio, cx, cy)

    if angle is not None:
        if debug:
            gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
            dbg = _build_debug(gray, circle_mask, binary, pointer_contour)
            dbg['angle'] = angle
            dbg['method'] = 'white_pointer'
            return angle, dbg
        return angle

    # 白色指针失败，尝试彩色把手方法
    angle, handle_contour, handle_mask = _try_color_handle(
        color_roi, circle_mask, circle_area,
        min_pointer_area_ratio, max_pointer_area_ratio, cx, cy)

    if angle is not None:
        if debug:
            gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
            dbg = _build_debug(gray, circle_mask, handle_mask, handle_contour)
            dbg['angle'] = angle
            dbg['method'] = 'color_handle'
            return angle, dbg
        return angle

    if debug:
        gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
        return None, _build_debug(gray, circle_mask,
                                  binary if binary is not None else np.zeros_like(circle_mask),
                                  None)
    return None


def _try_white_pointer(color_roi, circle_mask, radius, circle_area,
                       binary_thresh, min_area_ratio, max_area_ratio, cx, cy):
    """尝试用灰度二值化提取白色指针"""
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=circle_mask)

    # OTSU 自适应 + 固定阈值双策略
    _, binary_otsu = cv2.threshold(masked_gray, 0, 255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_otsu = cv2.bitwise_and(binary_otsu, circle_mask)

    _, binary_fixed = cv2.threshold(masked_gray, binary_thresh, 255,
                                    cv2.THRESH_BINARY)
    binary_fixed = cv2.bitwise_and(binary_fixed, circle_mask)

    otsu_white = cv2.countNonZero(binary_otsu)
    if otsu_white > circle_area * 0.4:
        binary = binary_fixed
    else:
        binary = binary_otsu

    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 轮廓筛选
    contour = _find_pointer_contour(
        binary, circle_area, min_area_ratio, max_area_ratio)
    if contour is None:
        return None, None, binary

    angle = _compute_angle_from_contour(contour, cx, cy)
    return angle, contour, binary


def _try_color_handle(color_roi, circle_mask, circle_area,
                      min_area_ratio, max_area_ratio, cx, cy):
    """
    尝试用自适应阈值提取暗色把手区域

    适用于红色/棕色旋转把手类旋钮，把手相对于底座有局部亮度差异。
    使用自适应阈值（反向）捕捉局部暗色区域（把手阴影/轮廓）。
    """
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=circle_mask)

    # 自适应阈值（反向）：提取局部暗色区域
    adaptive = cv2.adaptiveThreshold(
        masked_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5)
    adaptive = cv2.bitwise_and(adaptive, circle_mask)

    # 形态学处理：连接把手碎片
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    handle_mask = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel_small, iterations=1)
    handle_mask = cv2.morphologyEx(handle_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(handle_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, handle_mask

    # 把手特征：细长形状，长宽比 > 1.5，面积适中
    handle_min_area = circle_area * 0.02
    handle_max_area = circle_area * 0.5

    best_contour = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < handle_min_area or area > handle_max_area:
            continue

        rect = cv2.minAreaRect(cnt)
        box_w, box_h = rect[1]
        if min(box_w, box_h) < 1:
            continue
        aspect = max(box_w, box_h) / min(box_w, box_h)

        # 把手应该是细长形
        if aspect < 1.5:
            continue

        # 打分：偏好细长且面积大的轮廓
        score = aspect * math.sqrt(area)
        if score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is None:
        return None, None, handle_mask

    angle = _compute_angle_from_contour(best_contour, cx, cy)
    return angle, best_contour, handle_mask


def _find_pointer_contour(binary, circle_area, min_area_ratio, max_area_ratio):
    """从二值图中找到符合指针特征的轮廓"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = circle_area * min_area_ratio
    max_area = circle_area * max_area_ratio

    best_contour = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(cnt)
        box_w, box_h = rect[1]
        if min(box_w, box_h) < 1:
            continue
        aspect = max(box_w, box_h) / min(box_w, box_h)
        if aspect < 1.5:
            continue

        score = aspect * math.sqrt(area)
        if score > best_score:
            best_score = score
            best_contour = cnt

    return best_contour


def _compute_angle_from_contour(contour, cx, cy) -> float:
    """
    从轮廓计算指针角度

    策略：用 fitLine 获取方向向量，再用轮廓质心相对于旋钮中心的偏移消解 180° 歧义。
    """
    # fitLine 拟合方向
    line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(line[0][0]), float(line[1][0])

    # 轮廓质心（白色区域重心）
    M = cv2.moments(contour)
    if M['m00'] > 0:
        mcx = M['m10'] / M['m00']
        mcy = M['m01'] / M['m00']
    else:
        # fallback: 用轮廓边界框中心
        x, y, w, h = cv2.boundingRect(contour)
        mcx, mcy = x + w / 2, y + h / 2

    # 质心相对于旋钮中心的偏移方向
    dx = mcx - cx
    dy = mcy - cy

    # 用偏移方向消解 180° 歧义：
    # 指针从中心指向质心方向
    if vx * dx + vy * dy < 0:
        vx, vy = -vx, -vy

    # 计算角度：以 12 点钟方向 (向上) 为 0°，顺时针增加
    # 图像坐标系中 y 轴向下，所以 12 点钟方向对应 (0, -1)
    # atan2(vx, -vy) 将 (vx, vy) 映射到以 y 负方向为 0° 顺时针的角度
    angle_rad = math.atan2(vx, -vy)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360.0

    return angle_deg


def _build_debug(gray, circle_mask, binary, pointer_contour):
    """构造调试信息字典"""
    return {
        'gray': gray,
        'circle_mask': circle_mask,
        'binary': binary,
        'pointer_contour': pointer_contour,
    }


def draw_knob_angle(image, bbox, angle, color=(0, 255, 255), thickness=2):
    """
    在图像上绘制旋钮角度标注

    Args:
        image: 原始图像 (会被原地修改)
        bbox: (x1, y1, x2, y2) 旋钮边界框
        angle: 角度 (°)，12 点钟方向为 0°，顺时针
        color: 标注颜色
        thickness: 线条粗细
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r = min(x2 - x1, y2 - y1) // 3

    # 画指针方向线
    angle_rad = math.radians(angle)
    ex = int(cx + r * math.sin(angle_rad))
    ey = int(cy - r * math.cos(angle_rad))
    cv2.line(image, (cx, cy), (ex, ey), color, thickness, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), 3, color, -1, cv2.LINE_AA)

    # 标注角度数值
    label = f'{angle:.0f} deg'
    cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA)
