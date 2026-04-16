#!/usr/bin/env python3
"""
一键启动实时物体检测

使用 best.pt 模型 + Orbbec Gemini 336 相机，实时检测旋钮/指示灯并显示 3D 坐标。
所有参数从 config/yolov5s.yaml 读取，也可通过命令行覆盖。

用法:
    python run.py                        # 默认配置
    python run.py --weight best.pt       # 指定模型
    python run.py --camera realsense     # 切换相机
    python run.py --backend rknn         # 使用 NPU 推理
"""
import argparse
import time
import threading
import yaml
import random
import numpy as np
import cv2

from camera import create_backend
from depth_utils import (
    deproject_pixel_to_point, undistort_pixel,
    filter_depth, get_robust_depth, compute_panel_normal,
)
from knob_angle import estimate_knob_angle, draw_knob_angle


class AsyncCamera:
    """异步取帧：独立线程持续取帧，主线程直接读最新帧，不阻塞推理"""

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


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f.read(), Loader=yaml.SafeLoader)


def _try_set_names_from_pt(detector, pt_path):
    """尝试从 .pt 文件中提取类别名给 ONNX 检测器"""
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
            detector.colors = [[random.randint(0, 255) for _ in range(3)]
                               for _ in range(detector.class_num)]
            print(f'[INFO] 使用模型内置类别: {detector.class_names}')
    except Exception:
        pass


def create_detector(cfg, config_path, weight_override=None, backend_override=None):
    """根据配置创建检测器"""
    backend = backend_override or cfg.get('inference_backend', 'pytorch')
    weight = weight_override or cfg.get('weight', 'best.pt')

    if backend == 'rknn':
        from detector_rknn import YoloV5RKNN
        rknn_model = cfg.get('rknn_model', 'weights/yolov5s.rknn')
        print(f'[INFO] 推理后端: RKNN NPU ({rknn_model})')
        return YoloV5RKNN(rknn_model_path=rknn_model, config_path=config_path)

    if backend == 'onnx':
        from detector_onnx import YoloV5ORT
        onnx_path = weight.replace('.pt', '.onnx')
        threads = cfg.get('onnx_threads', 4)
        ort_det = YoloV5ORT(onnx_path=onnx_path, config_path=config_path, threads=threads)
        # 用模型权重推断类别名（onnx 没有内嵌 names，需要从 pt 加载）
        _try_set_names_from_pt(ort_det, weight)
        return ort_det

    # PyTorch 后端
    from utils.torch_utils import select_device, time_sync
    from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
    from utils.datasets import letterbox
    from models.experimental import attempt_load
    import torch.backends.cudnn as cudnn
    import torch

    print(f'[INFO] 推理后端: PyTorch, 模型: {weight}')

    class _Detector:
        def __init__(self):
            set_logging()
            device = select_device(cfg.get('device', 'cpu'))
            is_half = device.type != 'cpu'
            model = attempt_load(weight, map_location=device)
            input_size = check_img_size(cfg.get('input_size', 640), s=model.stride.max())
            if is_half:
                model.half()
            if torch.cuda.is_available():
                cudnn.benchmark = True
            img_torch = torch.zeros((1, 3, input_size, input_size), device=device)
            if device.type != 'cpu':
                model(img_torch.half() if is_half else img_torch)

            self.device = device
            self.model = model
            self.is_half = is_half
            self.input_size = input_size
            # 优先使用模型自带的类别名（避免 best.pt 与 config 类别不一致）
            if hasattr(model, 'names') and model.names:
                names = model.names
                if isinstance(names, dict):
                    self.class_names = [names[i] for i in sorted(names.keys())]
                else:
                    self.class_names = list(names)
                print(f'[INFO] 使用模型内置类别: {self.class_names}')
            else:
                self.class_names = cfg.get('class_name', [])
            self.class_num = len(self.class_names)
            self.conf_thresh = cfg['threshold']['confidence']
            self.iou_thresh = cfg['threshold']['iou']
            self.colors = [[random.randint(0, 255) for _ in range(3)]
                           for _ in range(self.class_num)]

        @torch.no_grad()
        def detect(self, img):
            img_resize = letterbox(img, new_shape=(self.input_size, self.input_size), auto=False)[0]
            img_arr = np.stack([img_resize], 0)
            img_arr = img_arr[:, :, :, ::-1].transpose(0, 3, 1, 2)
            img_arr = np.ascontiguousarray(img_arr)

            img_t = torch.from_numpy(img_arr).to(self.device)
            img_t = img_t.half() if self.is_half else img_t.float()
            img_t /= 255.0
            if img_t.ndimension() == 3:
                img_t = img_t.unsqueeze(0)

            pred = self.model(img_t, augment=False)[0]
            pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh)
            det = pred[0]

            canvas = np.copy(img)
            xyxy_list, conf_list, class_id_list = [], [], []
            if det is not None and len(det):
                det[:, :4] = scale_coords(img_arr.shape[2:], det[:, :4], img.shape).round()
                for *xyxy, conf, cls_id in reversed(det):
                    cls_id = int(cls_id)
                    xyxy_list.append(xyxy)
                    conf_list.append(conf)
                    class_id_list.append(cls_id)
                    label = f'{self.class_names[cls_id]} {conf:.2f}'
                    _plot_box(xyxy, canvas, label=label, color=self.colors[cls_id])
            return canvas, class_id_list, xyxy_list, conf_list

    return _Detector()


def _plot_box(x, img, color, label=None, thickness=2):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=thickness, lineType=cv2.LINE_AA)
    if label:
        tf = max(thickness - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=tf)[0]
        c2_text = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2_text, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, thickness / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description='一键启动实时物体检测')
    parser.add_argument('--config', default='config/yolov5s.yaml', help='配置文件路径')
    parser.add_argument('--weight', default='best.pt', help='模型权重路径')
    parser.add_argument('--camera', default=None, help='相机后端: orbbec / realsense')
    parser.add_argument('--backend', default=None, help='推理后端: pytorch / onnx / rknn')
    parser.add_argument('--no-depth', action='store_true', help='只检测不计算 3D 坐标')
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────
    cfg = load_config(args.config)

    # ── 初始化相机 ────────────────────────────────────────────────
    backend_type = args.camera or cfg.get('camera_backend', 'orbbec')
    cam_cfg = cfg.get('camera', {})
    cw = cam_cfg.get('color_width', 640)
    ch = cam_cfg.get('color_height', 480)
    dw = cam_cfg.get('depth_width', 640)
    dh = cam_cfg.get('depth_height', 480)
    fps = cam_cfg.get('fps', 30)

    print(f'[INFO] 相机: {backend_type}  分辨率: {cw}x{ch}@{fps}fps')
    raw_cam = create_backend(backend_type)
    raw_cam.initialize(cw, ch, dw, dh, fps)
    cam = AsyncCamera(raw_cam)
    print(f'[INFO] 异步取帧已启动')

    # ── 初始化检测器 ──────────────────────────────────────────────
    detector = create_detector(cfg, args.config,
                               weight_override=args.weight,
                               backend_override=args.backend)
    depth_scale = cam.get_depth_scale()
    compute_3d = not args.no_depth

    # ── 旋钮角度估计配置 ─────────────────────────────────────────
    angle_cfg = cfg.get('knob_angle', {})
    angle_enable = angle_cfg.get('enable', False)
    angle_binary_thresh = angle_cfg.get('binary_thresh', 180)
    angle_circle_mask = angle_cfg.get('circle_mask_ratio', 0.85)
    angle_knob_class = angle_cfg.get('knob_class', 'knob')

    # 面板法向量缓存（不需要每帧都算）
    panel_normal_cache = None
    panel_normal_frame_count = 0
    panel_normal_interval = 10  # 每 10 帧更新一次

    print(f'[INFO] 深度缩放因子: {depth_scale}')
    print(f'[INFO] 3D 坐标 + 面板法向量: {"开启" if compute_3d else "关闭"}')
    print(f'[INFO] 旋钮角度估计: {"开启" if angle_enable else "关闭"}')
    print('[INFO] 按 q 或 ESC 退出')
    print('=' * 50)

    # ── 主循环 ────────────────────────────────────────────────────
    try:
        while True:
            color_intrin, depth_intrin, color_image, depth_image = cam.get_aligned_frames()
            if color_intrin is None:
                continue
            if not color_image.any() or not depth_image.any():
                continue

            t_start = time.time()
            canvas, class_id_list, xyxy_list, conf_list = detector.detect(color_image)
            t_end = time.time()

            # ── 3D 坐标 + 面板法向量 ─────────────────────────────
            if compute_3d and xyxy_list:
                filtered_depth = filter_depth(depth_image, method='bilateral', kernel_size=5)

                # 面板法向量（每 N 帧更新一次）
                panel_normal_frame_count += 1
                if (panel_normal_cache is None or
                        panel_normal_frame_count % panel_normal_interval == 0):
                    result = compute_panel_normal(
                        color_image, filtered_depth, depth_intrin,
                        xyxy_list, depth_scale=depth_scale)
                    if result is not None:
                        panel_normal_cache = result

                # 显示面板法向量
                if panel_normal_cache is not None:
                    normal, centroid = panel_normal_cache
                    n_disp = np.round(normal, 3).tolist()
                    cv2.putText(canvas, f'panel_n: {n_disp}', (10, 80), 0, 0.6,
                                (0, 200, 255), thickness=2, lineType=cv2.LINE_AA)

                # 逐目标 3D 坐标
                for i, xyxy in enumerate(xyxy_list):
                    ux = int((xyxy[0] + xyxy[2]) / 2)
                    uy = int((xyxy[1] + xyxy[3]) / 2)

                    ux_u, uy_u = undistort_pixel(ux, uy, color_intrin)
                    ux_u, uy_u = int(ux_u), int(uy_u)

                    dis = get_robust_depth(filtered_depth, ux_u, uy_u,
                                           sample_radius=3, depth_scale=depth_scale)
                    xyz = deproject_pixel_to_point(depth_intrin, (ux_u, uy_u), dis)
                    xyz = np.round(xyz, 3).tolist()

                    cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                    cv2.putText(canvas, str(xyz), (ux + 20, uy + 10), 0, 0.6,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            # ── 旋钮角度估计 ─────────────────────────────────────
            if angle_enable and xyxy_list:
                for i, xyxy in enumerate(xyxy_list):
                    cls_name = detector.class_names[class_id_list[i]] \
                        if class_id_list[i] < len(detector.class_names) else ''
                    if cls_name != angle_knob_class:
                        continue
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    x2, y2 = int(xyxy[2]), int(xyxy[3])
                    roi = color_image[y1:y2, x1:x2]
                    angle = estimate_knob_angle(
                        roi,
                        binary_thresh=angle_binary_thresh,
                        circle_mask_ratio=angle_circle_mask,
                    )
                    if angle is not None:
                        draw_knob_angle(canvas, xyxy, angle)

            # ── FPS + 深度伪彩色 ─────────────────────────────────
            fps_val = int(1.0 / max(t_end - t_start, 1e-6))
            cv2.putText(canvas, f'FPS: {fps_val}', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            display = np.hstack((canvas, depth_colormap))

            cv2.namedWindow('detect', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow('detect', display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print('[INFO] 已退出')


if __name__ == '__main__':
    main()
