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

    # ── 1. 灰度 + 圆形 mask ───────────────────────────────────────
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)

    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), radius, 255, -1)

    masked_gray = cv2.bitwise_and(gray, gray, mask=circle_mask)

    # ── 2. 二值化提取白色指针 ─────────────────────────────────────
    # 先尝试 OTSU 自适应，如果效果不好 fallback 到固定阈值
    _, binary_otsu = cv2.threshold(masked_gray, 0, 255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_otsu = cv2.bitwise_and(binary_otsu, circle_mask)

    _, binary_fixed = cv2.threshold(masked_gray, binary_thresh, 255,
                                    cv2.THRESH_BINARY)
    binary_fixed = cv2.bitwise_and(binary_fixed, circle_mask)

    # 选择白色像素更少的那个（指针应该是小区域）
    otsu_white = cv2.countNonZero(binary_otsu)
    fixed_white = cv2.countNonZero(binary_fixed)
    circle_area = math.pi * radius * radius

    # 如果 OTSU 的白色区域太大（超过 40%），说明阈值太低，用固定阈值
    if otsu_white > circle_area * 0.4:
        binary = binary_fixed
    else:
        binary = binary_otsu

    # ── 3. 形态学去噪 ────────────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ── 4. 轮廓检测 + 筛选指针 ───────────────────────────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (None, _build_debug(gray, circle_mask, binary, None)) if debug else None

    # 按面积排序，找到符合指针特征的轮廓
    min_area = circle_area * min_pointer_area_ratio
    max_area = circle_area * max_pointer_area_ratio

    pointer_contour = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # 指针应该是长条形，长宽比 > 1.5
        rect = cv2.minAreaRect(cnt)
        box_w, box_h = rect[1]
        if min(box_w, box_h) < 1:
            continue
        aspect = max(box_w, box_h) / min(box_w, box_h)
        if aspect < 1.5:
            continue

        # 打分：偏好更细长、面积适中的轮廓
        score = aspect * math.sqrt(area)
        if score > best_score:
            best_score = score
            pointer_contour = cnt

    if pointer_contour is None:
        return (None, _build_debug(gray, circle_mask, binary, None)) if debug else None

    # ── 5. 计算指针角度 ──────────────────────────────────────────
    angle = _compute_angle_from_contour(pointer_contour, cx, cy)

    if debug:
        dbg = _build_debug(gray, circle_mask, binary, pointer_contour)
        dbg['angle'] = angle
        return angle, dbg

    return angle


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
