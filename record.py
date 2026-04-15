#!/usr/bin/env python3
"""
相机录制脚本 — 采集 YOLO 训练数据

功能:
  - 录制彩色视频 (.mp4)
  - 可选：同时保存深度视频 (.avi, 16bit 无损)
  - 可选：按键截图保存单帧（用于标注）
  - 显示实时画面 + 深度伪彩色

用法:
    python record.py                          # 默认录 color 视频
    python record.py --save-depth             # 同时录深度
    python record.py --no-video --snap-only   # 只截图不录视频
    python record.py -o data/scene1           # 指定输出目录
    python record.py --res 1280 720           # 指定分辨率

快捷键:
    s       截图保存单帧 (color + depth png)
    r       开始/暂停录制
    q/ESC   退出
"""
import argparse
import os
import time
import yaml
import numpy as np
import cv2

from camera import create_backend


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f.read(), Loader=yaml.SafeLoader)


def make_output_dir(base_dir):
    """创建输出目录，含 images/ 和 depth/ 子目录"""
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'depth'), exist_ok=True)
    return base_dir


def main():
    parser = argparse.ArgumentParser(description='相机录制 — 采集 YOLO 训练数据')
    parser.add_argument('--config', default='config/yolov5s.yaml', help='配置文件')
    parser.add_argument('-o', '--output', default='recordings', help='输出目录')
    parser.add_argument('--camera', default=None, help='相机后端: orbbec / realsense')
    parser.add_argument('--res', nargs=2, type=int, default=None,
                        metavar=('W', 'H'), help='分辨率，如 --res 1280 720')
    parser.add_argument('--fps', type=int, default=None, help='帧率')
    parser.add_argument('--save-depth', action='store_true', help='同时录制深度视频')
    parser.add_argument('--no-video', action='store_true', help='不录视频，只用截图模式')
    parser.add_argument('--snap-only', action='store_true', help='等同 --no-video')
    args = parser.parse_args()

    snap_only = args.no_video or args.snap_only

    # ── 加载配置 ──────────────────────────────────────────────────
    cfg = load_config(args.config)
    cam_cfg = cfg.get('camera', {})

    backend_type = args.camera or cfg.get('camera_backend', 'orbbec')
    if args.res:
        cw, ch = args.res
    else:
        cw = cam_cfg.get('color_width', 640)
        ch = cam_cfg.get('color_height', 480)
    dw, dh = cw, ch  # 深度流与彩色流同分辨率
    fps = args.fps or cam_cfg.get('fps', 30)

    # ── 初始化相机 ────────────────────────────────────────────────
    print(f'[INFO] 相机: {backend_type}  分辨率: {cw}x{ch}@{fps}fps')
    cam = create_backend(backend_type)
    cam.initialize(cw, ch, dw, dh, fps)

    # ── 准备输出目录 ──────────────────────────────────────────────
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = make_output_dir(os.path.join(args.output, timestamp))
    print(f'[INFO] 输出目录: {out_dir}')

    # ── 视频写入器 ────────────────────────────────────────────────
    color_writer = None
    depth_writer = None
    recording = False
    if not snap_only:
        recording = True  # 默认启动就开始录

    snap_count = 0
    frame_count = 0

    print()
    print('快捷键:  s=截图  r=开始/暂停录制  q/ESC=退出')
    print('=' * 50)

    try:
        while True:
            color_intrin, depth_intrin, color_image, depth_image = cam.get_aligned_frames()
            if color_intrin is None:
                continue
            if not color_image.any():
                continue

            frame_count += 1
            canvas = color_image.copy()

            # ── 录制状态 ──────────────────────────────────────────
            if recording and color_writer is None:
                video_path = os.path.join(out_dir, 'color.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                color_writer = cv2.VideoWriter(video_path, fourcc, fps, (cw, ch))
                print(f'[REC] 开始录制: {video_path}')

                if args.save_depth:
                    depth_video_path = os.path.join(out_dir, 'depth.avi')
                    # 用 FFV1 无损编码保存 16bit 深度，灰度单通道
                    depth_fourcc = cv2.VideoWriter_fourcc(*'FFV1')
                    depth_writer = cv2.VideoWriter(
                        depth_video_path, depth_fourcc, fps, (dw, dh), isColor=False)
                    print(f'[REC] 深度录制: {depth_video_path}')

            if recording and color_writer is not None:
                color_writer.write(color_image)
                if depth_writer is not None and depth_image is not None:
                    # 16bit depth 转 8bit 用于 VideoWriter（或用 png 序列更好）
                    depth_writer.write((depth_image >> 4).astype(np.uint8))

            # ── OSD 显示 ──────────────────────────────────────────
            # 录制状态指示
            if recording:
                cv2.circle(canvas, (cw - 30, 30), 10, (0, 0, 255), -1)
                cv2.putText(canvas, 'REC', (cw - 75, 37), 0, 0.6,
                            (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(canvas, 'PAUSE', (cw - 90, 37), 0, 0.6,
                            (128, 128, 128), 2, cv2.LINE_AA)

            cv2.putText(canvas, f'Frame: {frame_count}  Snap: {snap_count}',
                        (10, 30), 0, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # 深度伪彩色
            if depth_image is not None:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                display = np.hstack((canvas, depth_colormap))
            else:
                display = canvas

            cv2.namedWindow('record', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow('record', display)
            key = cv2.waitKey(1) & 0xFF

            # ── 按键处理 ──────────────────────────────────────────
            if key in (ord('q'), 27):
                break

            elif key == ord('s'):
                # 截图：保存 color png + depth png
                snap_count += 1
                name = f'{timestamp}_{snap_count:04d}'
                color_path = os.path.join(out_dir, 'images', f'{name}.jpg')
                cv2.imwrite(color_path, color_image)
                print(f'[SNAP] {color_path}')

                if depth_image is not None:
                    depth_path = os.path.join(out_dir, 'depth', f'{name}.png')
                    cv2.imwrite(depth_path, depth_image)

            elif key == ord('r'):
                if snap_only:
                    print('[INFO] 截图模式，录制不可用')
                else:
                    recording = not recording
                    if recording:
                        print('[REC] 继续录制')
                    else:
                        print('[REC] 暂停录制')

    except KeyboardInterrupt:
        pass
    finally:
        if color_writer is not None:
            color_writer.release()
            print(f'[INFO] 彩色视频已保存')
        if depth_writer is not None:
            depth_writer.release()
            print(f'[INFO] 深度视频已保存')
        cam.stop()
        cv2.destroyAllWindows()

        print(f'[INFO] 共 {frame_count} 帧, {snap_count} 张截图')
        print(f'[INFO] 输出目录: {out_dir}')


if __name__ == '__main__':
    main()
