"""
相机后端抽象基类与统一内参数据结构

支持 RealSense D435i / Orbbec Gemini 336 等深度相机的统一接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class CameraIntrinsics:
    """统一相机内参数据结构"""
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0  # RealSense 中对应 ppx
    cy: float = 0.0  # RealSense 中对应 ppy
    width: int = 0
    height: int = 0
    coeffs: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # coeffs 顺序: [k1, k2, p1, p2, k3]


class CameraBackend(ABC):
    """深度相机后端抽象基类"""

    @abstractmethod
    def initialize(self, color_width: int = 640, color_height: int = 480,
                   depth_width: int = 640, depth_height: int = 480,
                   fps: int = 30) -> None:
        """
        初始化相机管线

        Args:
            color_width: 彩色流宽度
            color_height: 彩色流高度
            depth_width: 深度流宽度
            depth_height: 深度流高度
            fps: 帧率
        """
        pass

    @abstractmethod
    def get_aligned_frames(self) -> Tuple[CameraIntrinsics, CameraIntrinsics,
                                          np.ndarray, np.ndarray]:
        """
        获取对齐后的彩色帧和深度帧

        Returns:
            (color_intrin, depth_intrin, color_image, depth_image)
            - color_intrin: 彩色相机内参
            - depth_intrin: 深度相机内参
            - color_image: BGR 彩色图 (H, W, 3) uint8
            - depth_image: 深度图 (H, W) uint16
        """
        pass

    @abstractmethod
    def get_depth_scale(self) -> float:
        """
        获取深度缩放因子（将原始深度值转换为米）

        Returns:
            深度缩放因子，例如 0.001 表示原始值单位为 mm
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """停止相机管线并释放资源"""
        pass
