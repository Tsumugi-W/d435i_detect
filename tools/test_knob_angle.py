#!/usr/bin/env python3
"""
旋钮角度估计离线验证脚本

从录制的视频中自动检测黑色圆形旋钮区域，调用 knob_angle 模块估计角度，
在画面上标注结果。仅依赖 OpenCV + numpy，可在 macOS 上直接运行。

用法:
    # 处理单个视频，显示实时窗口
    python tools/test_knob_angle.py /path/to/color.mp4

    # 处理视频并保存标注结果
    python tools/test_knob_angle.py /path/to/color.mp4 --save output.mp4

    # 处理目录下所有录制（自动找 color.mp4）
    python tools/test_knob_angle.py /path/to/recordings/

    # 调试模式：显示中间处理步骤
    python tools/test_knob_angle.py /path/to/color.mp4 --debug

快捷键:
    空格  暂停/继续
    d     切换调试显示
    s     截图保存当前帧
    q/ESC 退出
"""
import argparse
import os
import sys
import math

import cv2
import numpy as np

# 将项目根目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from knob_angle import estimate_knob_angle, draw_knob_angle


def detect_knobs_by_color(image, min_radius=18, max_radius=55):
    """
    用传统 CV 检测黑色圆形旋钮区域（不依赖 YOLO 模型）

    策略：
    1. HoughCircles 检测圆形候选
    2. 验证圆内颜色为深色 + 低饱和度
    3. 验证圆内存在亮色指针条（面积比例合理）
    4. 验证圆边缘有清晰的亮/暗对比（旋钮有金属或白色边框）
    5. 验证圆形轮廓的圆度

    Args:
        image: BGR 图像
        min_radius: 最小旋钮半径 (px)，面板旋钮在 640x480 下通常 20~45px
        max_radius: 最大旋钮半径 (px)

    Returns:
        list of (x1, y1, x2, y2) bbox
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # HoughCircles 检测 — 收紧 param2 减少弱圆
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_radius * 2.5,
        param1=100,
        param2=50,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    bboxes = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for circle in circles[0]:
        cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
        ri = int(r)

        # 边界检查（留足够余量）
        margin = ri + 5
        if cx - margin < 0 or cy - margin < 0 or cx + margin >= w or cy + margin >= h:
            continue

        # ── 检查 1：圆内核心区域为深色 ────────────────────────────
        core_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(core_mask, (cx, cy), int(r * 0.6), 255, -1)
        mean_val = cv2.mean(hsv, mask=core_mask)
        mean_s, mean_v = mean_val[1], mean_val[2]

        # 旋钮表面：黑色/深灰，低亮度 + 低饱和度
        if mean_v > 100:
            continue
        if mean_s > 80:  # 排除绿色按钮等高饱和度圆形
            continue

        # ── 检查 2：圆内存在白色指针条 ────────────────────────────
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(inner_mask, (cx, cy), int(r * 0.8), 255, -1)
        roi_gray = cv2.bitwise_and(gray, gray, mask=inner_mask)
        bright_pixels = cv2.countNonZero(
            cv2.threshold(roi_gray, 170, 255, cv2.THRESH_BINARY)[1]
        )
        total_pixels = cv2.countNonZero(inner_mask)
        bright_ratio = bright_pixels / max(total_pixels, 1)

        # 指针占圆面积 3%~20%
        if bright_ratio < 0.03 or bright_ratio > 0.20:
            continue

        # ── 检查 3：边缘对比度 ────────────────────────────────────
        # 旋钮有银色/白色金属边框，边缘内外应有明显亮度跳变
        ring_outer = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring_outer, (cx, cy), int(r * 1.15), 255, -1)
        ring_inner = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring_inner, (cx, cy), int(r * 0.85), 255, -1)
        ring_mask = cv2.subtract(ring_outer, ring_inner)
        ring_mean = cv2.mean(gray, mask=ring_mask)[0]
        core_mean_gray = cv2.mean(gray, mask=core_mask)[0]
        edge_contrast = abs(ring_mean - core_mean_gray)

        # 边缘对比度至少 30（旋钮边框 vs 黑色表面）
        if edge_contrast < 25:
            continue

        # 通过所有验证，生成 bbox
        pad = int(r * 0.15)
        x1 = max(0, cx - ri - pad)
        y1 = max(0, cy - ri - pad)
        x2 = min(w, cx + ri + pad)
        y2 = min(h, cy + ri + pad)
        bboxes.append((x1, y1, x2, y2))

    return bboxes


def process_frame(image, show_debug=False):
    """
    处理单帧：检测旋钮 + 估计角度

    Returns:
        canvas: 标注后的图像
        results: list of (bbox, angle)
        debug_images: dict of debug visualizations (if show_debug)
    """
    canvas = image.copy()
    results = []
    debug_images = {}

    bboxes = detect_knobs_by_color(image)

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        if show_debug:
            angle, dbg = estimate_knob_angle(roi, debug=True)
            debug_images[f'knob_{i}'] = dbg
        else:
            angle = estimate_knob_angle(roi)

        if angle is not None:
            draw_knob_angle(canvas, bbox, angle)
            results.append((bbox, angle))
            # 画 bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
        else:
            # 检测到旋钮但无法估计角度
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(canvas, '?', (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return canvas, results, debug_images


def build_debug_panel(debug_images, panel_height=200):
    """将调试中间结果拼成一行"""
    panels = []
    for name, dbg in debug_images.items():
        if dbg is None:
            continue
        parts = []

        # 二值化结果
        if 'binary' in dbg and dbg['binary'] is not None:
            binary_vis = cv2.cvtColor(dbg['binary'], cv2.COLOR_GRAY2BGR)
            # 画轮廓
            if dbg.get('pointer_contour') is not None:
                cv2.drawContours(binary_vis, [dbg['pointer_contour']], -1,
                                 (0, 0, 255), 2)
            parts.append(binary_vis)

        # 灰度 + 圆形 mask
        if 'gray' in dbg and dbg['gray'] is not None:
            gray_vis = cv2.cvtColor(dbg['gray'], cv2.COLOR_GRAY2BGR)
            parts.append(gray_vis)

        if not parts:
            continue

        # 统一高度
        resized = []
        for p in parts:
            ph, pw = p.shape[:2]
            scale = panel_height / ph
            resized.append(cv2.resize(p, (int(pw * scale), panel_height)))

        row = np.hstack(resized)
        # 加标题
        label = name
        if dbg.get('angle') is not None:
            label += f'  {dbg["angle"]:.0f} deg'
        cv2.putText(row, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        panels.append(row)

    if not panels:
        return None

    # 统一宽度后纵向拼接
    max_w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.zeros((p.shape[0], max_w - p.shape[1], 3), dtype=np.uint8)
            p = np.hstack([p, pad])
        padded.append(p)

    return np.vstack(padded)


def find_videos(path):
    """找到所有可处理的视频路径"""
    if os.path.isfile(path):
        return [path]

    videos = []
    if os.path.isdir(path):
        # 直接是录制目录（含 color.mp4）
        mp4 = os.path.join(path, 'color.mp4')
        if os.path.isfile(mp4):
            return [mp4]
        # 是 recordings 父目录
        for entry in sorted(os.listdir(path)):
            sub = os.path.join(path, entry)
            if os.path.isdir(sub):
                mp4 = os.path.join(sub, 'color.mp4')
                if os.path.isfile(mp4):
                    videos.append(mp4)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description='旋钮角度估计离线验证',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='快捷键: 空格=暂停  d=调试  s=截图  q=退出',
    )
    parser.add_argument('input', help='视频文件或录制目录路径')
    parser.add_argument('--save', default=None, help='保存标注视频的路径')
    parser.add_argument('--debug', action='store_true', help='显示调试中间结果')
    parser.add_argument('--skip', type=int, default=0, help='跳过前 N 帧')
    parser.add_argument('--step', type=int, default=1, help='每隔 N 帧处理一次')
    args = parser.parse_args()

    videos = find_videos(args.input)
    if not videos:
        print(f'[ERROR] 未找到视频: {args.input}')
        sys.exit(1)

    print(f'[INFO] 找到 {len(videos)} 个视频')

    for video_path in videos:
        print(f'\n[INFO] 处理: {video_path}')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'[ERROR] 无法打开: {video_path}')
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'[INFO] {w}x{h} @ {fps:.0f}fps, {total} frames')

        writer = None
        if args.save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

        paused = False
        show_debug = args.debug
        frame_idx = 0
        snap_count = 0

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_idx <= args.skip:
                    continue
                if (frame_idx - args.skip) % args.step != 0:
                    continue

                canvas, results, debug_images = process_frame(frame, show_debug)

                # FPS / 进度信息
                progress = f'{frame_idx}/{total}'
                cv2.putText(canvas, progress, (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(canvas, f'knobs: {len(results)}', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if writer:
                    writer.write(canvas)

            # 显示
            cv2.imshow('Knob Angle Test', canvas)

            if show_debug and debug_images:
                debug_panel = build_debug_panel(debug_images)
                if debug_panel is not None:
                    cv2.imshow('Debug', debug_panel)

            wait_ms = 1 if not paused else 50
            key = cv2.waitKey(wait_ms) & 0xFF

            if key in (ord('q'), 27):
                cap.release()
                if writer:
                    writer.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused
            elif key == ord('d'):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow('Debug')
            elif key == ord('s'):
                snap_name = f'knob_snap_{snap_count:04d}.jpg'
                cv2.imwrite(snap_name, canvas)
                print(f'[SNAP] {snap_name}')
                snap_count += 1

        cap.release()
        if writer:
            writer.release()
            print(f'[INFO] 已保存: {args.save}')

    cv2.destroyAllWindows()
    print('\n[INFO] 完成')


if __name__ == '__main__':
    main()
