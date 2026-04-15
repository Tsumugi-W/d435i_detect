"""
应用配置加载模块

从 config/yolov5s.yaml 中读取相机后端、分辨率、推理后端等配置，
并提供全局相机实例的初始化。
"""
import yaml
from camera import create_backend

# 全局相机实例（由 init_camera() 初始化）
camera = None

# 已加载的配置字典
config = {}

CONFIG_PATH = 'config/yolov5s.yaml'


def load_config(config_path=None):
    """加载 YAML 配置文件"""
    global config
    path = config_path or CONFIG_PATH
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


def init_camera(config_path=None):
    """根据配置初始化相机后端"""
    global camera
    if not config:
        load_config(config_path)

    backend_type = config.get('camera_backend', 'orbbec')
    cam_cfg = config.get('camera', {})
    color_w = cam_cfg.get('color_width', 640)
    color_h = cam_cfg.get('color_height', 480)
    depth_w = cam_cfg.get('depth_width', 640)
    depth_h = cam_cfg.get('depth_height', 480)
    fps = cam_cfg.get('fps', 30)

    print(f'[INFO] 相机后端: {backend_type}')
    print(f'[INFO] 分辨率: color={color_w}x{color_h}, depth={depth_w}x{depth_h}, fps={fps}')

    camera = create_backend(backend_type)
    camera.initialize(color_w, color_h, depth_w, depth_h, fps)
    return camera


def create_detector(config_path=None):
    """
    根据配置创建检测器（PyTorch 或 RKNN）

    Returns:
        检测器实例，具有 detect(img) 方法
    """
    if not config:
        load_config(config_path)

    path = config_path or CONFIG_PATH
    inference_backend = config.get('inference_backend', 'pytorch')

    if inference_backend == 'rknn':
        from detector_rknn import YoloV5RKNN
        rknn_model = config.get('rknn_model', 'weights/yolov5s.rknn')
        print(f'[INFO] 推理后端: RKNN NPU ({rknn_model})')
        return YoloV5RKNN(rknn_model_path=rknn_model, config_path=path)
    else:
        # 延迟导入，避免没用到时也依赖 torch
        print(f'[INFO] 推理后端: PyTorch CPU/GPU')
        return None  # 返回 None 表示使用脚本内置的 YoloV5 类
