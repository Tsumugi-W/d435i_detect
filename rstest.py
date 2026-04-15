'''
by yzh 2022.2.13
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
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose
from collections import deque

from depth_utils import (
    deproject_pixel_to_point, undistort_pixel,
    filter_depth, get_robust_depth,
)
import app_config

# 相机实例（由 main() 中 app_config.init_camera() 初始化）
camera = None


def get_aligned_images():
    color_intrin, depth_intrin, color_image, depth_image = camera.get_aligned_frames()
    if color_intrin is None:
        return None, None, None, None, None
    return color_intrin, depth_intrin, color_image, depth_image, depth_image


class MultiFrameTracker:
    """
    多帧融合追踪器，对多帧检测结果进行加权平均
    """

    def __init__(self, window_size=5, decay_factor=0.8):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.history = deque(maxlen=window_size)

    def update(self, detections):
        self.history.append(detections)

        if len(self.history) < 2:
            return detections

        fused_detections = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det

            historical_coords = []
            weights = []

            for frame_idx, frame_dets in enumerate(self.history):
                if i < len(frame_dets):
                    hist_det = frame_dets[i]
                    historical_coords.append(hist_det[:4])
                    weight = self.decay_factor ** (len(self.history) - frame_idx - 1)
                    weights.append(weight)

            if historical_coords:
                weights = np.array(weights)
                weights = weights / weights.sum()
                fused_coords = np.average(historical_coords, axis=0, weights=weights)
                x1, y1, x2, y2 = fused_coords
                fused_detections.append((x1, y1, x2, y2))
            else:
                fused_detections.append(det)

        return fused_detections


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


class DetectionPublisher(Node):
    def __init__(self):
        super().__init__('yolov5_detection_publisher')

        self.detection_pub = self.create_publisher(
            Detection3DArray,
            'detection_3d',
            10
        )

        self.coords_pub = self.create_publisher(
            String,
            'detection_coords',
            10
        )

        print("[INFO] YoloV5目标检测-程序启动")
        print("[INFO] 开始模型加载")
        # 根据配置选择推理后端
        rknn_detector = app_config.create_detector()
        if rknn_detector is not None:
            self.model = rknn_detector
        else:
            self.model = YoloV5(yolov5_yaml_path='config/yolov5s.yaml')
        print("[INFO] 完成模型加载")

        self.tracker = MultiFrameTracker(window_size=5, decay_factor=0.8)
        print("[INFO] 多帧融合追踪器已初始化（窗口大小：5）")

        self.timer = self.create_timer(0.033, self.detection_callback)  # 30Hz

    def detection_callback(self):
        try:
            intr, depth_intrin, color_image, depth_image, _ = get_aligned_images()
            if intr is None or not depth_image.any() or not color_image.any():
                return

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            t_start = time.time()
            canvas, class_id_list, xyxy_list, conf_list = self.model.detect(
                color_image)

            xyxy_list = self.tracker.update(xyxy_list)

            t_end = time.time()

            # 深度滤波
            filtered_depth = filter_depth(depth_image, method='bilateral', kernel_size=5)

            depth_scale = camera.get_depth_scale()
            camera_xyz_list = []
            if xyxy_list:
                for i in range(len(xyxy_list)):
                    ux = int((xyxy_list[i][0]+xyxy_list[i][2])/2)
                    uy = int((xyxy_list[i][1]+xyxy_list[i][3])/2)

                    # 去畸变
                    ux_undistorted, uy_undistorted = undistort_pixel(ux, uy, intr)
                    ux_undistorted = int(ux_undistorted)
                    uy_undistorted = int(uy_undistorted)

                    # 多点采样深度
                    dis = get_robust_depth(filtered_depth, ux_undistorted, uy_undistorted,
                                          sample_radius=3, depth_scale=depth_scale)

                    # 反投影为 3D 坐标
                    camera_xyz = deproject_pixel_to_point(
                        depth_intrin, (ux_undistorted, uy_undistorted), dis)
                    camera_xyz = np.round(np.array(camera_xyz), 3).tolist()
                    cv2.circle(canvas, (ux, uy), 4, (255, 255, 255), 5)
                    cv2.putText(canvas, str(camera_xyz), (ux+20, uy+10), 0, 1,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    camera_xyz_list.append(camera_xyz)

            self.publish_detections(camera_xyz_list, class_id_list, conf_list)

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
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.get_logger().error(f'Error in detection: {str(e)}')

    def publish_detections(self, camera_xyz_list, class_id_list, conf_list):
        '''发布检测结果到ROS2话题'''
        detection_array = Detection3DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera_link'

        for i, xyz in enumerate(camera_xyz_list):
            detection = Detection3D()
            detection.bbox.center.position.x = float(xyz[0])
            detection.bbox.center.position.y = float(xyz[1])
            detection.bbox.center.position.z = float(xyz[2])

            if i < len(class_id_list):
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(class_id_list[i])
                hypothesis.hypothesis.score = float(conf_list[i])
                detection.results.append(hypothesis)

            detection_array.detections.append(detection)

        self.detection_pub.publish(detection_array)

        coords_msg = String()
        coords_msg.data = str(camera_xyz_list)
        self.coords_pub.publish(coords_msg)

        self.get_logger().info(f'Published {len(camera_xyz_list)} detections')


def main(args=None):
    global camera
    app_config.load_config()
    camera = app_config.init_camera()

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
