'''
by yzh 2022.2.13
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

import pyrealsense2 as rs
import math
import yaml
import argparse
import os
import time
import numpy as np
import sys

import cv2

# ROS2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose
# PyTorch
# YoloV5-PyTorch

pipeline = rs.pipeline()  # 定义流程pipeline
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)  # 流程开始
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)


def undistort_pixel(u, v, intr, iterations=5):
    """
    将畸变的像素坐标转换为理想坐标（去畸变）

    使用 Brown-Conrady 畸变模型的反演

    Args:
        u, v: 实际像素坐标
        intr: RealSense 内参对象，包含 fx, fy, ppx, ppy, coeffs
        iterations: 迭代次数（通常 3-5 次收敛）

    Returns:
        u_ideal, v_ideal: 去畸变后的理想像素坐标
    """
    # 相机内参
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy

    # 畸变系数 [k1, k2, p1, p2, k3]
    coeffs = intr.coeffs
    if len(coeffs) >= 5:
        k1, k2, p1, p2, k3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]
    else:
        # 如果系数不足，用 0 填充
        k1 = coeffs[0] if len(coeffs) > 0 else 0
        k2 = coeffs[1] if len(coeffs) > 1 else 0
        p1 = coeffs[2] if len(coeffs) > 2 else 0
        p2 = coeffs[3] if len(coeffs) > 3 else 0
        k3 = coeffs[4] if len(coeffs) > 4 else 0

    # 转换为归一化坐标
    x = (u - cx) / fx
    y = (v - cy) / fy

    # 迭代求解理想坐标（牛顿法反演）
    for _ in range(iterations):
        r2 = x**2 + y**2

        # 计算径向和切向畸变
        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        dx = 2*p1*x*y + p2*(r2 + 2*x**2)
        dy = p1*(r2 + 2*y**2) + 2*p2*x*y

        # 反演：从畸变坐标恢复理想坐标
        x_new = (x - dx) / radial
        y_new = (y - dy) / radial

        # 检查收敛
        if abs(x_new - x) < 1e-6 and abs(y_new - y) < 1e-6:
            break

        x, y = x_new, y_new

    # 转换回像素坐标
    u_ideal = x * fx + cx
    v_ideal = y * fy + cy

    return u_ideal, v_ideal


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

    ############### 相机参数的获取 #######################
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }'''

    # 保存内参到本地
    # with open('./intrinsics.json', 'w') as fp:
    #json.dump(camera_parameters, fp)
    #######################################################

    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图

    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame


class YoloV5:
    def __init__(self, yolov5_yaml_path='config/yolov5s.yaml'):
        '''初始化'''
        # 载入配置文件
        with open(yolov5_yaml_path, 'r', encoding='utf-8') as f:
            self.yolov5 = yaml.load(f.read(), Loader=yaml.SafeLoader)
        # 随机生成每个类别的颜色
        self.colors = [[np.random.randint(0, 255) for _ in range(
            3)] for class_id in range(self.yolov5['class_num'])]
        # 模型初始化
        self.init_model()

    @torch.no_grad()
    def init_model(self):
        '''模型初始化'''
        # 设置日志输出
        set_logging()
        # 选择计算设备
        device = select_device(self.yolov5['device'])
        # 如果是GPU则使用半精度浮点数 F16
        is_half = device.type != 'cpu'
        # 载入模型
        model = attempt_load(
            self.yolov5['weight'], map_location=device)  # 载入全精度浮点数的模型
        input_size = check_img_size(
            self.yolov5['input_size'], s=model.stride.max())  # 检查模型的尺寸
        if is_half:
            model.half()  # 将模型转换为半精度
        # 设置BenchMark，加速固定图像的尺寸的推理
        cudnn.benchmark = True  # set True to speed up constant image size inference
        # 图像缓冲区初始化
        img_torch = torch.zeros(
            (1, 3, self.yolov5['input_size'], self.yolov5['input_size']), device=device)  # init img
        # 创建模型
        # run once
        _ = model(img_torch.half()
                  if is_half else img) if device.type != 'cpu' else None
        self.is_half = is_half  # 是否开启半精度
        self.device = device  # 计算设备
        self.model = model  # Yolov5模型
        self.img_torch = img_torch  # 图像缓冲区

    def preprocessing(self, img):
        '''图像预处理'''
        # 图像缩放
        # 注: auto一定要设置为False -> 图像的宽高不同
        img_resize = letterbox(img, new_shape=(
            self.yolov5['input_size'], self.yolov5['input_size']), auto=False)[0]
        # print("img resize shape: {}".format(img_resize.shape))
        # 增加一个维度
        img_arr = np.stack([img_resize], 0)
        # 图像转换 (Convert) BGR格式转换为RGB
        # 转换为 bs x 3 x 416 x
        # 0(图像i), 1(row行), 2(列), 3(RGB三通道)
        # ---> 0, 3, 1, 2
        # BGR to RGB, to bsx3x416x416
        img_arr = img_arr[:, :, :, ::-1].transpose(0, 3, 1, 2)
        # 数值归一化
        # img_arr =  img_arr.astype(np.float32) / 255.0
        # 将数组在内存的存放地址变成连续的(一维)， 行优先
        # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        # https://zhuanlan.zhihu.com/p/59767914
        img_arr = np.ascontiguousarray(img_arr)
        return img_arr

    @torch.no_grad()
    def detect(self, img, canvas=None, view_img=True):
        '''模型预测'''
        # 图像预处理
        img_resize = self.preprocessing(img)  # 图像缩放
        self.img_torch = torch.from_numpy(img_resize).to(self.device)  # 图像格式转换
        self.img_torch = self.img_torch.half(
        ) if self.is_half else self.img_torch.float()  # 格式转换 uint8-> 浮点数
        self.img_torch /= 255.0  # 图像归一化
        if self.img_torch.ndimension() == 3:
            self.img_torch = self.img_torch.unsqueeze(0)
        # 模型推理
        t1 = time_sync()
        pred = self.model(self.img_torch, augment=False)[0]
        # pred = self.model_trt(self.img_torch, augment=False)[0]
        # NMS 非极大值抑制
        pred = non_max_suppression(pred, self.yolov5['threshold']['confidence'],
                                   self.yolov5['threshold']['iou'], classes=None, agnostic=False)
        t2 = time_sync()
        # print("推理时间: inference period = {}".format(t2 - t1))
        # 获取检测结果
        det = pred[0]
        gain_whwh = torch.tensor(img.shape)[[1, 0, 1, 0]]  # [w, h, w, h]

        if view_img and canvas is None:
            canvas = np.copy(img)
        xyxy_list = []
        conf_list = []
        class_id_list = []
        if det is not None and len(det):
            # 画面中存在目标对象
            # 将坐标信息恢复到原始图像的尺寸
            det[:, :4] = scale_coords(
                img_resize.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, class_id in reversed(det):
                class_id = int(class_id)
                xyxy_list.append(xyxy)
                conf_list.append(conf)
                class_id_list.append(class_id)
                if view_img:
                    # 绘制矩形框与标签
                    label = '%s %.2f' % (
                        self.yolov5['class_name'][class_id], conf)
                    self.plot_one_box(
                        xyxy, canvas, label=label, color=self.colors[class_id], line_thickness=3)
        return canvas, class_id_list, xyxy_list, conf_list

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        ''''绘制矩形框+标签'''
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class DetectionPublisher(Node):
    def __init__(self):
        super().__init__('yolov5_detection_publisher')
        
        # 创建发布者
        self.detection_pub = self.create_publisher(
            Detection3DArray, 
            'detection_3d', 
            10
        )
        
        # 创建简单坐标发布者 (可选)
        self.coords_pub = self.create_publisher(
            String,
            'detection_coords',
            10
        )
        
        print("[INFO] YoloV5目标检测-程序启动")
        print("[INFO] 开始YoloV5模型加载")
        self.model = YoloV5(yolov5_yaml_path='config/yolov5s.yaml')
        print("[INFO] 完成YoloV5模型加载")
        
        self.timer = self.create_timer(0.033, self.detection_callback)  # 30Hz
        
    def detection_callback(self):
        try:
            # Wait for a coherent pair of frames: depth and color
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()  # 获取对齐的图像与相机内参
            if not depth_image.any() or not color_image.any():
                continue
            # Convert images to numpy arrays
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
                depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            
            # Show images

            t_start = time.time()  # 开始计时
            # YoloV5 目标检测
            canvas, class_id_list, xyxy_list, conf_list = self.model.detect(
                color_image)

            t_end = time.time()  # 结束计时\
            #canvas = np.hstack((canvas, depth_colormap))
            #print(class_id_list)

            camera_xyz_list=[]
            if xyxy_list:
                for i in range(len(xyxy_list)):
                    ux = int((xyxy_list[i][0]+xyxy_list[i][2])/2)  # 计算像素坐标系的x
                    uy = int((xyxy_list[i][1]+xyxy_list[i][3])/2)  # 计算像素坐标系的y

                    # 去畸变：将畸变的像素坐标转换为理想坐标
                    ux_undistorted, uy_undistorted = undistort_pixel(ux, uy, intr)
                    ux_undistorted = int(ux_undistorted)
                    uy_undistorted = int(uy_undistorted)

                    # 使用去畸变后的坐标获取深度值
                    dis = aligned_depth_frame.get_distance(ux_undistorted, uy_undistorted)

                    # 使用去畸变后的坐标进行 3D 转换
                    camera_xyz = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, (ux_undistorted, uy_undistorted), dis)  # 计算相机坐标系的xyz
                    camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                    camera_xyz = camera_xyz.tolist()
                    cv2.circle(canvas, (ux,uy), 4, (255, 255, 255), 5)#标出中心点
                    cv2.putText(canvas, str(camera_xyz), (ux+20, uy+10), 0, 1,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)#标出坐标
                    camera_xyz_list.append(camera_xyz)
            
            # 发布ROS2消息
            self.publish_detections(camera_xyz_list, class_id_list, conf_list)

            # 添加fps显示
            fps = int(1.0 / (t_end - t_start))
            cv2.putText(canvas, text="FPS: {}".format(fps), org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                        lineType=cv2.LINE_AA, color=(0, 0, 0))
            cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                            cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.imshow('detection', canvas)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
        except Exception as e:
            self.get_logger().error(f'Error in detection: {str(e)}')
    
    def publish_detections(self, camera_xyz_list, class_id_list, conf_list):
        '''发布检测结果到ROS2话题'''
        # 发布标准Detection3DArray消息
        detection_array = Detection3DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera_link'
        
        for i, xyz in enumerate(camera_xyz_list):
            detection = Detection3D()
            
            # 设置位置
            detection.bbox.center.position.x = float(xyz[0])
            detection.bbox.center.position.y = float(xyz[1])
            detection.bbox.center.position.z = float(xyz[2])
            
            # 设置类别和置信度
            if i < len(class_id_list):
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(class_id_list[i])
                hypothesis.hypothesis.score = float(conf_list[i])
                detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)
        
        self.detection_pub.publish(detection_array)
        
        # 同时发布简单字符串格式 (可选)
        coords_msg = String()
        coords_msg.data = str(camera_xyz_list)
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
