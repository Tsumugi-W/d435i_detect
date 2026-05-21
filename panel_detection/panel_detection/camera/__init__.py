"""
相机后端模块

用法:
    from camera import create_backend
    camera = create_backend('orbbec')  # 或 'realsense'
    camera.initialize(640, 480, 640, 480, 30)
"""
from .base import CameraBackend, CameraIntrinsics


def create_backend(backend_type: str) -> CameraBackend:
    """
    根据类型创建相机后端实例

    Args:
        backend_type: 'realsense' 或 'orbbec'

    Returns:
        CameraBackend 实例（未初始化，需调用 initialize()）
    """
    backend_type = backend_type.lower().strip()

    if backend_type == 'realsense':
        from .realsense import RealSenseBackend
        return RealSenseBackend()
    elif backend_type == 'orbbec':
        from .orbbec import OrbbecBackend
        return OrbbecBackend()
    else:
        raise ValueError(
            f"不支持的相机后端: '{backend_type}'，"
            f"可选: 'realsense', 'orbbec'")
