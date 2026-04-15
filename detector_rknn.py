"""
RKNN NPU 推理器 — 在 RK3588 上使用 NPU 加速 YOLOv5 推理

用法:
    detector = YoloV5RKNN('weights/yolov5s.rknn', 'config/yolov5s.yaml')
    canvas, class_id_list, xyxy_list, conf_list = detector.detect(color_image)

要求:
    - RK3588 平台
    - rknn-toolkit-lite2 已安装
    - 模型已通过 tools/convert_to_rknn.py 转换为 .rknn 格式
"""
import numpy as np
import cv2
import yaml
import random

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False


# ── 后处理常量 ──────────────────────────────────────────────────
OBJ_THRESH = 0.25
NMS_THRESH = 0.45


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """图像等比缩放 + padding"""
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


def _nms_boxes(boxes, scores, nms_thresh):
    """单类 NMS"""
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
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
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)


def _process_output(outputs, img_shape, ori_shape, ratio, pad,
                    anchors, conf_thresh, nms_thresh):
    """
    解码 RKNN 输出的三层特征图，执行 NMS

    Args:
        outputs: RKNN 推理输出 list，三个特征图
        img_shape: 推理输入尺寸 (h, w)
        ori_shape: 原始图像尺寸 (h, w)
        ratio: letterbox 缩放比
        pad: letterbox padding (dw, dh)
        anchors: anchor 尺寸列表
        conf_thresh: 置信度阈值
        nms_thresh: NMS IoU 阈值

    Returns:
        (boxes, scores, class_ids) — 均为 numpy 数组
    """
    strides = [8, 16, 32]
    all_boxes, all_scores, all_class_ids = [], [], []

    for i, feat in enumerate(outputs):
        # feat shape: (1, na, h, w, nc+5) 或 (1, h, w, na*(nc+5))
        # 具体 shape 取决于导出方式，此处兼容常见格式
        if feat.ndim == 5:
            # (1, na, h, w, nc+5)
            feat = feat[0]
        elif feat.ndim == 4:
            # (1, h, w, na*(nc+5)) -> 需 reshape
            bs, fh, fw, _ = feat.shape
            na = len(anchors[i])
            feat = feat.reshape(bs, fh, fw, na, -1)[0].transpose(2, 0, 1, 3)
            # -> (na, fh, fw, nc+5)
        else:
            continue

        na, fh, fw, nc5 = feat.shape
        nc = nc5 - 5

        # 生成网格
        grid_y, grid_x = np.meshgrid(np.arange(fh), np.arange(fw), indexing='ij')

        for a in range(na):
            data = feat[a]  # (fh, fw, nc+5)
            box_xy = _sigmoid(data[..., :2])
            box_wh = _sigmoid(data[..., 2:4])
            obj_conf = _sigmoid(data[..., 4])
            cls_conf = _sigmoid(data[..., 5:])

            # 解码 bbox
            cx = (box_xy[..., 0] * 2 - 0.5 + grid_x) * strides[i]
            cy = (box_xy[..., 1] * 2 - 0.5 + grid_y) * strides[i]
            bw = (box_wh[..., 0] * 2) ** 2 * anchors[i][a][0]
            bh = (box_wh[..., 1] * 2) ** 2 * anchors[i][a][1]

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            # 筛选高置信度
            for c in range(nc):
                scores = obj_conf * cls_conf[..., c]
                mask = scores > conf_thresh
                if not mask.any():
                    continue
                boxes = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1)
                # 还原到原始图像坐标
                boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
                boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio
                # clip
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, ori_shape[1])
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, ori_shape[0])

                all_boxes.append(boxes)
                all_scores.append(scores[mask])
                all_class_ids.append(np.full(mask.sum(), c, dtype=np.int32))

    if not all_boxes:
        return np.array([]), np.array([]), np.array([])

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_class_ids = np.concatenate(all_class_ids, axis=0)

    # 按类别做 NMS
    final_boxes, final_scores, final_ids = [], [], []
    for c in np.unique(all_class_ids):
        mask = all_class_ids == c
        c_boxes = all_boxes[mask]
        c_scores = all_scores[mask]
        keep = _nms_boxes(c_boxes, c_scores, nms_thresh)
        final_boxes.append(c_boxes[keep])
        final_scores.append(c_scores[keep])
        final_ids.append(np.full(len(keep), c, dtype=np.int32))

    return (np.concatenate(final_boxes),
            np.concatenate(final_scores),
            np.concatenate(final_ids))


class YoloV5RKNN:
    """
    RKNN NPU 加速的 YOLOv5 检测器

    接口与 YoloV5 (PyTorch) 类保持一致，可直接替换使用
    """

    # YOLOv5s 默认 anchors（与模型导出时一致）
    DEFAULT_ANCHORS = [
        [[10, 13], [16, 30], [33, 23]],       # P3/8
        [[30, 61], [62, 45], [59, 119]],       # P4/16
        [[116, 90], [156, 198], [373, 326]],   # P5/32
    ]

    def __init__(self, rknn_model_path='weights/yolov5s.rknn',
                 config_path='config/yolov5s.yaml'):
        if not HAS_RKNN:
            raise ImportError(
                "rknn-toolkit-lite2 未安装，请从 Rockchip SDK 安装 rknnlite")

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f.read(), Loader=yaml.SafeLoader)

        self.input_size = self.config.get('input_size', 640)
        self.class_names = self.config.get('class_name', [])
        self.conf_thresh = self.config['threshold']['confidence']
        self.nms_thresh = self.config['threshold']['iou']
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(self.config.get('class_num', 2))]

        # 初始化 RKNN
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(rknn_model_path)
        if ret != 0:
            raise RuntimeError(f'加载 RKNN 模型失败: {rknn_model_path}')

        # 使用三核 NPU（RK3588 专属，RK3566/3568 使用 NPU_CORE_0）
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError('RKNN 运行时初始化失败')

        print(f'[INFO] RKNN 模型加载成功: {rknn_model_path}')

    def detect(self, img):
        """
        执行检测，接口与 PyTorch 版 YoloV5.detect() 一致

        Args:
            img: BGR 彩色图 (H, W, 3) uint8

        Returns:
            (canvas, class_id_list, xyxy_list, conf_list)
        """
        ori_shape = img.shape[:2]

        # 前处理
        img_lb, ratio, pad = _letterbox(
            img, new_shape=(self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

        # NPU 推理（RKNN 默认接受 uint8 RGB 输入，内部做量化）
        outputs = self.rknn.inference(inputs=[img_rgb])

        # 后处理
        boxes, scores, class_ids = _process_output(
            outputs,
            img_shape=(self.input_size, self.input_size),
            ori_shape=ori_shape,
            ratio=ratio,
            pad=pad,
            anchors=self.DEFAULT_ANCHORS,
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
        )

        canvas = np.copy(img)
        xyxy_list, conf_list, class_id_list = [], [], []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            xyxy = [x1, y1, x2, y2]
            xyxy_list.append(xyxy)
            conf_list.append(float(scores[i]))
            class_id_list.append(int(class_ids[i]))

            label = '%s %.2f' % (self.class_names[int(class_ids[i])], scores[i])
            self._plot_box(xyxy, canvas, label=label,
                           color=self.colors[int(class_ids[i])])

        return canvas, class_id_list, xyxy_list, conf_list

    def _plot_box(self, x, img, color, label=None, line_thickness=3):
        c1 = (int(x[0]), int(x[1]))
        c2 = (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=line_thickness,
                      lineType=cv2.LINE_AA)
        if label:
            tf = max(line_thickness - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3,
                                     thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0,
                        line_thickness / 3, [225, 255, 255],
                        thickness=tf, lineType=cv2.LINE_AA)

    def release(self):
        """释放 RKNN 资源"""
        if self.rknn:
            self.rknn.release()
