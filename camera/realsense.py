"""
Intel RealSense D435i 相机后端

将 pyrealsense2 API 封装为统一的 CameraBackend 接口
"""
import numpy as np
from .base import CameraBackend, CameraIntrinsics

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


class RealSenseBackend(CameraBackend):
    """Intel RealSense D435i 后端"""

    def __init__(self):
        if not HAS_REALSENSE:
            raise ImportError(
                "pyrealsense2 未安装，请执行: pip install pyrealsense2")
        self.pipeline = None
        self.align = None
        self.profile = None
        self._depth_scale = 0.001  # 默认值，initialize() 中会更新

    def initialize(self, color_width=848, color_height=480,
                   depth_width=848, depth_height=480, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, depth_width, depth_height,
                             rs.format.z16, fps)
        config.enable_stream(rs.stream.color, color_width, color_height,
                             rs.format.bgr8, fps)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # 获取实际的深度缩放因子
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

    def get_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return None, None, None, None

        # 提取内参并转换为统一格式
        color_intrin = self._convert_intrinsics(
            color_frame.profile.as_video_stream_profile().intrinsics)
        depth_intrin = self._convert_intrinsics(
            aligned_depth_frame.profile.as_video_stream_profile().intrinsics)

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_intrin, depth_intrin, color_image, depth_image

    def get_depth_scale(self):
        return self._depth_scale

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

    @staticmethod
    def _convert_intrinsics(rs_intrin) -> CameraIntrinsics:
        """将 pyrealsense2 内参对象转换为统一格式"""
        return CameraIntrinsics(
            fx=rs_intrin.fx,
            fy=rs_intrin.fy,
            cx=rs_intrin.ppx,  # RealSense ppx -> 统一 cx
            cy=rs_intrin.ppy,  # RealSense ppy -> 统一 cy
            width=rs_intrin.width,
            height=rs_intrin.height,
            coeffs=list(rs_intrin.coeffs),
        )
