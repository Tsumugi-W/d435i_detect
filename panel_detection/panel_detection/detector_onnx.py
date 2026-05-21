"""
ONNX Runtime 推理器 — CPU 加速 YOLOv5 推理

相比 PyTorch CPU 推理，OnnxRuntime 在 RK3588 上快 2-3 倍。

用法:
    detector = YoloV5ORT('best.onnx', 'config/yolov5s.yaml')
    canvas, class_id_list, xyxy_list, conf_list = detector.detect(color_image)
"""
import numpy as np
import cv2
import yaml
import random

import onnxruntime as ort


def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def _xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= iou_thresh)[0] + 1]
    return np.array(keep)


def _postprocess(output, ori_shape, input_size, conf_thresh, iou_thresh):
    """解码 YOLOv5 ONNX 输出并做 NMS"""
    # output shape: (1, N, 5+nc) — 已经 decoded by model
    pred = output[0]  # (N, 5+nc)

    # 过滤低置信度
    obj_conf = pred[:, 4]
    mask = obj_conf > conf_thresh
    pred = pred[mask]
    if len(pred) == 0:
        return [], [], []

    # class conf = obj_conf * cls_conf
    cls_scores = pred[:, 5:] * pred[:, 4:5]
    class_ids = cls_scores.argmax(axis=1)
    confidences = cls_scores[np.arange(len(cls_scores)), class_ids]

    # 二次过滤
    mask2 = confidences > conf_thresh
    pred = pred[mask2]
    class_ids = class_ids[mask2]
    confidences = confidences[mask2]

    if len(pred) == 0:
        return [], [], []

    # xywh -> xyxy
    boxes = _xywh2xyxy(pred[:, :4])

    # 还原到原始图像坐标
    oh, ow = ori_shape
    gain = min(input_size / oh, input_size / ow)
    pad_w = (input_size - ow * gain) / 2
    pad_h = (input_size - oh * gain) / 2
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / gain
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ow)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, oh)

    # 按类别 NMS
    final_boxes, final_scores, final_ids = [], [], []
    for c in np.unique(class_ids):
        c_mask = class_ids == c
        c_boxes = boxes[c_mask]
        c_scores = confidences[c_mask]
        keep = _nms(c_boxes, c_scores, iou_thresh)
        final_boxes.append(c_boxes[keep])
        final_scores.append(c_scores[keep])
        final_ids.append(np.full(len(keep), c, dtype=np.int32))

    if not final_boxes:
        return [], [], []

    all_boxes = np.concatenate(final_boxes)
    all_scores = np.concatenate(final_scores)
    all_ids = np.concatenate(final_ids)

    # 跨类别 NMS：同一区域只保留置信度最高的结果
    cross_keep = _nms(all_boxes, all_scores, iou_thresh)
    return all_boxes[cross_keep], all_scores[cross_keep], all_ids[cross_keep]


class YoloV5ORT:
    """OnnxRuntime 加速的 YOLOv5 检测器"""

    def __init__(self, onnx_path='best.onnx', config_path='config/yolov5s.yaml',
                 threads=4):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)

        self.input_size = cfg.get('input_size', 640)
        self.conf_thresh = cfg['threshold']['confidence']
        self.iou_thresh = cfg['threshold']['iou']

        # 创建 session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path, opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

        # 类别名：尝试从模型元数据获取，否则从 config 读取
        self.class_names = cfg.get('class_name', [])
        self.class_num = len(self.class_names)
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(self.class_num)]

        # 预热
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})

        print(f'[INFO] ONNX Runtime 推理器就绪: {onnx_path} ({threads} threads)')

    def detect(self, img):
        ori_shape = img.shape[:2]

        # 前处理
        img_lb, _, _ = _letterbox(img, (self.input_size, self.input_size))
        inp = img_lb[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        inp = np.ascontiguousarray(inp, dtype=np.float32) / 255.0
        inp = inp[np.newaxis, ...]  # add batch dim

        # 推理
        output = self.session.run(None, {self.input_name: inp})[0]

        # 后处理
        boxes, scores, class_ids = _postprocess(
            output, ori_shape, self.input_size,
            self.conf_thresh, self.iou_thresh)

        canvas = np.copy(img)
        xyxy_list, conf_list, class_id_list = [], [], []

        if len(boxes) > 0:
            for i in range(len(boxes)):
                xyxy = boxes[i].tolist()
                xyxy_list.append(xyxy)
                conf_list.append(float(scores[i]))
                class_id_list.append(int(class_ids[i]))

                label = f'{self.class_names[int(class_ids[i])]} {scores[i]:.2f}'
                self._plot_box(xyxy, canvas, label=label,
                               color=self.colors[int(class_ids[i])])

        return canvas, class_id_list, xyxy_list, conf_list

    def _plot_box(self, x, img, color, label=None, thickness=2):
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=thickness, lineType=cv2.LINE_AA)
        if label:
            tf = max(thickness - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=tf)[0]
            c2_t = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2_t, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, thickness / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
