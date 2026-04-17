#!/usr/bin/env python3
"""
离线视频测试脚本 — 用 recordings 中的视频验证检测 + 角度估计效果

功能:
  - 逐帧运行 YOLOv5 检测
  - 对 knob 类目标估计旋钮角度
  - 显示检测结果 + 角度标注
  - 可选保存输出视频
  - 统计检测性能（FPS、检出率）

用法:
    python tools/test_video.py                                    # 自动找最新录制
    python tools/test_video.py -i recordings/20260415_175610/color.mp4
    python tools/test_video.py -i recordings/20260415_175610/color.mp4 --save
    python tools/test_video.py -i recordings/20260415_175610/color.mp4 --no-show  # 无 GUI 纯跑数
"""
import argparse
import glob
import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run import create_detector, load_config, _try_set_names_from_pt, _plot_box
from knob_angle import estimate_knob_angle, draw_knob_angle


def find_latest_video():
    """自动查找 recordings/ 下最新的 color.mp4"""
    videos = sorted(glob.glob('recordings/*/color.mp4'))
    if not videos:
        print('[ERROR] 未找到录制视频，请先用 record.py 录制')
        sys.exit(1)
    return videos[-1]


def main():
    parser = argparse.ArgumentParser(description='离线视频测试')
    parser.add_argument('-i', '--input', default=None, help='输入视频路径')
    parser.add_argument('--config', default='config/yolov5s.yaml', help='配置文件')
    parser.add_argument('--weight', default='best.pt', help='模型权重')
    parser.add_argument('--backend', default='onnx', help='推理后端: onnx / pytorch')
    parser.add_argument('--save', action='store_true', help='保存输出视频')
    parser.add_argument('--no-show', action='store_true', help='不显示画面（纯跑数统计）')
    parser.add_argument('--skip', type=int, default=0, help='跳过前 N 帧')
    parser.add_argument('--max-frames', type=int, default=0, help='最多处理 N 帧（0=全部）')
    args = parser.parse_args()

    video_path = args.input or find_latest_video()
    print(f'[INFO] 输入视频: {video_path}')

    # ── 加载配置和检测器 ──────────────────────────────────────────
    cfg = load_config(args.config)
    detector = create_detector(cfg, args.config,
                               weight_override=args.weight,
                               backend_override=args.backend)

    # 角度估计配置
    angle_cfg = cfg.get('knob_angle', {})
    angle_enable = angle_cfg.get('enable', True)
    binary_thresh = angle_cfg.get('binary_thresh', 180)
    circle_mask_ratio = angle_cfg.get('circle_mask_ratio', 0.85)
    knob_class = angle_cfg.get('knob_class', 'knob')

    class_names = getattr(detector, 'class_names', cfg.get('class_name', []))
    print(f'[INFO] 类别: {class_names}')
    print(f'[INFO] 角度估计: {"开启" if angle_enable else "关闭"}')

    # ── 打开视频 ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'[ERROR] 无法打开视频: {video_path}')
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[INFO] 视频: {vid_w}x{vid_h} @ {vid_fps:.0f}fps, {total_frames} 帧')

    # 跳帧
    if args.skip > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.skip)
        print(f'[INFO] 跳过前 {args.skip} 帧')

    # ── 输出视频 ──────────────────────────────────────────────────
    writer = None
    if args.save:
        out_path = video_path.replace('color.mp4', 'result.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, vid_fps, (vid_w, vid_h))
        print(f'[INFO] 输出视频: {out_path}')

    # ── 统计变量 ──────────────────────────────────────────────────
    frame_idx = 0
    det_frames = 0      # 有检出的帧数
    total_dets = 0       # 总检出数
    total_knobs = 0
    total_angles = 0     # 成功估计角度数
    fps_list = []

    max_frames = args.max_frames if args.max_frames > 0 else float('inf')

    print()
    print('处理中...')
    if not args.no_show:
        print('按 q/ESC 退出，空格键暂停')

    paused = False

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.time()

        # 检测
        canvas, class_id_list, xyxy_list, conf_list = detector.detect(frame)
        t1 = time.time()

        n_det = len(class_id_list)
        if n_det > 0:
            det_frames += 1
        total_dets += n_det

        # 角度估计
        n_knob_this = 0
        n_angle_this = 0
        if angle_enable and xyxy_list:
            for i, xyxy in enumerate(xyxy_list):
                cls_name = class_names[class_id_list[i]] \
                    if class_id_list[i] < len(class_names) else ''
                if cls_name != knob_class:
                    continue
                n_knob_this += 1
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                x2, y2 = int(xyxy[2]), int(xyxy[3])
                roi = frame[y1:y2, x1:x2]
                angle = estimate_knob_angle(
                    roi, binary_thresh=binary_thresh,
                    circle_mask_ratio=circle_mask_ratio)
                if angle is not None:
                    n_angle_this += 1
                    draw_knob_angle(canvas, xyxy, angle)

        total_knobs += n_knob_this
        total_angles += n_angle_this

        fps = 1.0 / max(t1 - t0, 1e-6)
        fps_list.append(fps)

        # OSD
        cv2.putText(canvas, f'FPS: {fps:.0f}', (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f'Frame: {frame_idx}/{total_frames}', (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(canvas, f'Det: {n_det}  Knob: {n_knob_this}  Angle: {n_angle_this}',
                    (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 进度
        if frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            avg_fps = np.mean(fps_list[-100:])
            print(f'  帧 {frame_idx}/{total_frames} ({pct:.0f}%) '
                  f'FPS={avg_fps:.1f} 检出={total_dets}')

        if writer is not None:
            writer.write(canvas)

        if not args.no_show:
            cv2.namedWindow('test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow('test', canvas)
            wait_ms = 0 if paused else 1
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                paused = not paused

    # ── 统计报告 ──────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    avg_fps = np.mean(fps_list) if fps_list else 0
    det_rate = det_frames / frame_idx * 100 if frame_idx > 0 else 0
    angle_rate = total_angles / total_knobs * 100 if total_knobs > 0 else 0

    print()
    print('=' * 50)
    print(f'  测试报告')
    print('=' * 50)
    print(f'  视频: {video_path}')
    print(f'  处理帧数: {frame_idx}')
    print(f'  平均 FPS: {avg_fps:.1f}')
    print(f'  总检出数: {total_dets}')
    print(f'  检出帧率: {det_rate:.1f}% ({det_frames}/{frame_idx} 帧有检出)')
    print(f'  旋钮总数: {total_knobs}')
    print(f'  角度估计成功率: {angle_rate:.1f}% ({total_angles}/{total_knobs})')
    if args.save:
        print(f'  输出视频: {video_path.replace("color.mp4", "result.mp4")}')
    print('=' * 50)


if __name__ == '__main__':
    main()
