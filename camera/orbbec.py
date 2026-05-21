"""
奥比中光 Gemini 336 相机后端

将 pyorbbecsdk API 封装为统一的 CameraBackend 接口
"""
import numpy as np
from .base import CameraBackend, CameraIntrinsics

try:
    from pyorbbecsdk import (
        Pipeline, Config, OBSensorType, OBStreamType, OBFormat,
        AlignFilter, FrameSet, VideoFrame,
    )
    HAS_ORBBEC = True
except ImportError:
    HAS_ORBBEC = False


def _frame_to_bgr(color_frame: 'VideoFrame') -> np.ndarray:
    """
    将 Orbbec 彩色帧转换为 BGR numpy 数组

    Gemini 336 可能输出 MJPG / YUYV / RGB / BGR 等格式，
    此函数统一转换为 OpenCV 的 BGR 格式。
    """
    import cv2

    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    if fmt == OBFormat.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.YUYV:
        yuyv = data.reshape((height, width, 2))
        bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
    elif fmt == OBFormat.NV21:
        nv21 = data.reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(nv21, cv2.COLOR_YUV2BGR_NV21)
    elif fmt == OBFormat.NV12:
        nv12 = data.reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    elif fmt == OBFormat.RGB:
        rgb = data.reshape((height, width, 3))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.UYVY:
        uyvy = data.reshape((height, width, 2))
        bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    else:
        # BGR 或其他未知格式，直接 reshape
        bgr = data.reshape((height, width, 3))

    return bgr


class OrbbecBackend(CameraBackend):
    """奥比中光 Gemini 336 后端"""

    def __init__(self):
        if not HAS_ORBBEC:
            raise ImportError(
                "pyorbbecsdk 未安装，请参考奥比中光 SDK 文档安装")
        self.pipeline = None
        self.align_filter = None
        self._depth_scale = 0.001  # 默认值，initialize() 中会更新
        self._color_intrinsics = None
        self._depth_intrinsics = None

    def initialize(self, color_width=640, color_height=480,
                   depth_width=640, depth_height=480, fps=30):
        self.pipeline = Pipeline()
        config = Config()

        # 配置彩色流
        color_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR)
        color_profile = self._find_profile(
            color_profiles, color_width, color_height, fps)
        if color_profile is None:
            color_profile = color_profiles.get_default_video_stream_profile()
            print(f"[WARN] 未找到 {color_width}x{color_height}@{fps} 彩色流，"
                  f"使用默认 profile")
        config.enable_stream(color_profile)

        # 配置深度流
        depth_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR)
        depth_profile = self._find_profile(
            depth_profiles, depth_width, depth_height, fps)
        if depth_profile is None:
            depth_profile = depth_profiles.get_default_video_stream_profile()
            print(f"[WARN] 未找到 {depth_width}x{depth_height}@{fps} 深度流，"
                  f"使用默认 profile")
        config.enable_stream(depth_profile)

        # 启动管线
        self.pipeline.enable_frame_sync()
        self.pipeline.start(config)

        # 深度对齐到彩色
        self.align_filter = AlignFilter(
            align_to_stream=OBStreamType.COLOR_STREAM)

        # 获取相机内参（一次性获取，运行中不变）
        camera_param = self.pipeline.get_camera_param()
        self._color_intrinsics = self._convert_intrinsics(
            camera_param.rgb_intrinsic, camera_param.rgb_distortion)
        self._depth_intrinsics = self._convert_intrinsics(
            camera_param.depth_intrinsic, camera_param.depth_distortion)

    def get_aligned_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(1000)
        except Exception:
            return None, None, None, None
        if frames is None:
            return None, None, None, None

        # 确保 color 和 depth 帧都存在后再做对齐
        if frames.get_color_frame() is None or frames.get_depth_frame() is None:
            return None, None, None, None

        # 深度对齐到彩色
        try:
            aligned = self.align_filter.process(frames)
        except Exception:
            return None, None, None, None
        if aligned is None:
            return None, None, None, None

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None, None, None, None

        # 更新深度缩放因子
        self._depth_scale = depth_frame.get_depth_scale()

        # 转换为 numpy
        color_image = _frame_to_bgr(color_frame)
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_image = depth_data.reshape(
            (depth_frame.get_height(), depth_frame.get_width()))

        return (self._color_intrinsics, self._depth_intrinsics,
                color_image, depth_image)

    def get_depth_scale(self):
        return self._depth_scale

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

    @staticmethod
    def _find_profile(profile_list, width, height, fps):
        """在 profile 列表中查找匹配分辨率和帧率的 profile"""
        count = profile_list.get_count()
        # 精确匹配：分辨率 + 帧率
        for i in range(count):
            p = profile_list.get_stream_profile_by_index(i)
            vp = p.as_video_stream_profile()
            if (vp.get_width() == width and
                    vp.get_height() == height and
                    vp.get_fps() == fps):
                return p
        # 放宽帧率限制再找一次
        for i in range(count):
            p = profile_list.get_stream_profile_by_index(i)
            vp = p.as_video_stream_profile()
            if (vp.get_width() == width and
                    vp.get_height() == height):
                return p
        return None

    @staticmethod
    def _convert_intrinsics(ob_intrin, ob_distortion=None) -> CameraIntrinsics:
        """将 Orbbec 内参/畸变转换为统一格式"""
        coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        if ob_distortion is not None:
            coeffs = [
                ob_distortion.k1,
                ob_distortion.k2,
                ob_distortion.p1,
                ob_distortion.p2,
                ob_distortion.k3,
            ]
        return CameraIntrinsics(
            fx=ob_intrin.fx,
            fy=ob_intrin.fy,
            cx=ob_intrin.cx,
            cy=ob_intrin.cy,
            width=ob_intrin.width,
            height=ob_intrin.height,
            coeffs=coeffs,
        )
