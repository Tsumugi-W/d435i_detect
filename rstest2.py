'''
by yzh 2022.2.13
简化版：无 ROS2、无法向量估计
适配 RealSense D435i / Orbbec Gemini 336 统一相机后端
'''
# 导入依赖
import random
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, strip_optimizer, set_logging)
from utils.datasets import LoadStreams, LoadImages, letterbox
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch

import math
import yaml
import argparse
import os
import time
import numpy as np
import sys

import cv2

from depth_utils import deproject_pixel_to_point
import app_config

# 相机实例（由 main 块中 app_config.init_camera() 初始化）
camera = None


def get_aligned_images():
    color_intrin, depth_intrin, color_image, depth_image = camera.get_aligned_frames()
    if color_intrin is None:
        return None, None, None, None, None

    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit))

    return color_intrin, depth_intrin, color_image, depth_image, depth_image


class YoloV5:
    def __init__(self, yolov5_yaml_path='config/yolov5s.yaml'):
        '''初始化'''
        with open(yolov5_yaml_path, 'r', encoding='utf-8') as f:
            self.yolov5 = yaml.load(f.read(), Loader=yaml.SafeLoader)
        self.colors = [[np.random.randint(0, 255) for _ in range(
            3)] for class_id in range(self.yolov5['class_num'])]
        self.init_model()

    @torch.no_grad()
    def init_model(self):
        '''模型初始化'''
        set_logging()
        device = select_device(self.yolov5['device'])
        is_half = device.type != 'cpu'
        model = attempt_load(
            self.yolov5['weight'], map_location=device)
        input_size = check_img_size(
            self.yolov5['input_size'], s=model.stride.max())
        if is_half:
            model.half()
        if torch.cuda.is_available():
            cudnn.benchmark = True
        img_torch = torch.zeros(
            (1, 3, self.yolov5['input_size'], self.yolov5['input_size']), device=device)
        _ = model(img_torch.half()
                  if is_half else img_torch) if device.type != 'cpu' else None
        self.is_half = is_half
        self.device = device
        self.model = model
        self.img_torch = img_torch

    def preprocessing(self, img):
        '''图像预处理'''
        img_resize = letterbox(img, new_shape=(
            self.yolov5['input_size'], self.yolov5['input_size']), auto=False)[0]
        img_arr = np.stack([img_resize], 0)
        img_arr = img_arr[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img_arr = np.ascontiguousarray(img_arr)
        return img_arr

    @torch.no_grad()
    def detect(self, img, canvas=None, view_img=True):
        '''模型预测'''
        img_resize = self.preprocessing(img)
        self.img_torch = torch.from_numpy(img_resize).to(self.device)
        self.img_torch = self.img_torch.half(
        ) if self.is_half else self.img_torch.float()
        self.img_torch /= 255.0
        if self.img_torch.ndimension() == 3:
            self.img_torch = self.img_torch.unsqueeze(0)
        t1 = time_sync()
        pred = self.model(self.img_torch, augment=False)[0]
        pred = non_max_suppression(pred, self.yolov5['threshold']['confidence'],
                                   self.yolov5['threshold']['iou'], classes=None, agnostic=False)
        t2 = time_sync()
        det = pred[0]
        gain_whwh = torch.tensor(img.shape)[[1, 0, 1, 0]]

        if view_img and canvas is None:
            canvas = np.copy(img)
        xyxy_list = []
        conf_list = []
        class_id_list = []
        if det is not None and len(det):
            det[:, :4] = scale_coords(
                img_resize.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, class_id in reversed(det):
                class_id = int(class_id)
                xyxy_list.append(xyxy)
                conf_list.append(conf)
                class_id_list.append(class_id)
                if view_img:
                    label = '%s %.2f' % (
                        self.yolov5['class_name'][class_id], conf)
                    self.plot_one_box(
                        xyxy, canvas, label=label, color=self.colors[class_id], line_thickness=3)
        return canvas, class_id_list, xyxy_list, conf_list

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        ''''绘制矩形框+标签'''
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    app_config.load_config()
    camera = app_config.init_camera()

    print("[INFO] YoloV5目标检测-程序启动")
    print("[INFO] 开始模型加载")
    rknn_detector = app_config.create_detector()
    if rknn_detector is not None:
        model = rknn_detector
    else:
        model = YoloV5(yolov5_yaml_path='config/yolov5s.yaml')
    print("[INFO] 完成模型加载")

    depth_scale = camera.get_depth_scale()

    try:
        while True:
            intr, depth_intrin, color_image, depth_image, _ = get_aligned_images()
            if intr is None or not depth_image.any() or not color_image.any():
                continue

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            t_start = time.time()
            canvas, class_id_list, xyxy_list, conf_list = model.detect(
                color_image)

            t_end = time.time()

            camera_xyz_list = []
            if xyxy_list:
                for i in range(len(xyxy_list)):
                    ux = int((xyxy_list[i][0]+xyxy_list[i][2])/2)
                    uy = int((xyxy_list[i][1]+xyxy_list[i][3])/2)
                    # 从深度图直接读取深度值
                    uy_clamped = max(0, min(depth_image.shape[0] - 1, uy))
                    ux_clamped = max(0, min(depth_image.shape[1] - 1, ux))
                    dis = depth_image[uy_clamped, ux_clamped] * depth_scale
                    camera_xyz = deproject_pixel_to_point(
                        depth_intrin, (ux, uy), dis)
                    camera_xyz = np.round(np.array(camera_xyz), 3).tolist()
                    cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                    cv2.putText(canvas, str(camera_xyz), (ux+20, uy+10), 0, 1,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    camera_xyz_list.append(camera_xyz)

            fps = int(1.0 / (t_end - t_start))
            cv2.putText(canvas, text="FPS: {}".format(fps), org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                        lineType=cv2.LINE_AA, color=(0, 0, 0))
            cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                            cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow('detection', canvas)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        camera.stop()
