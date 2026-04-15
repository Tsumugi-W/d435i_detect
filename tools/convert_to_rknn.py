"""
YOLOv5 模型转换脚本: .pt -> .onnx -> .rknn

在 x86 主机上运行（rknn-toolkit2 仅支持 x86），
转换完成后将 .rknn 文件部署到 RK3588。

用法:
    python tools/convert_to_rknn.py \
        --pt weights/yolov5s.pt \
        --output weights/yolov5s.rknn \
        --img-size 640 \
        --platform rk3588

依赖:
    - torch, torchvision (导出 ONNX)
    - onnx (ONNX 验证)
    - rknn-toolkit2 (ONNX -> RKNN 转换，需 x86 环境)

注意:
    推荐使用 airockchip 修改版 YOLOv5 导出 ONNX，
    该版本会去掉模型尾部的后处理层，使 NPU 推理更高效。
    仓库: https://github.com/airockchip/yolov5
"""
import argparse
import os
import sys


def export_onnx(pt_path, onnx_path, img_size=640):
    """Step 1: PyTorch .pt -> ONNX"""
    import torch

    print(f'[1/3] 导出 ONNX: {pt_path} -> {onnx_path}')

    # 加载模型
    model = torch.load(pt_path, map_location='cpu')
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    model = model.float().eval()

    # dummy input
    dummy = torch.zeros(1, 3, img_size, img_size)

    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=12,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=None,
    )
    print(f'    ONNX 导出完成: {onnx_path}')

    # 验证
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print('    ONNX 模型验证通过')
    except ImportError:
        print('    [WARN] onnx 未安装，跳过验证')


def convert_rknn(onnx_path, rknn_path, platform='rk3588', img_size=640,
                 quantize=True, dataset_path=None):
    """Step 2: ONNX -> RKNN"""
    try:
        from rknn.api import RKNN
    except ImportError:
        print('[ERROR] rknn-toolkit2 未安装。')
        print('  请在 x86 环境安装: pip install rknn-toolkit2')
        print('  参考: https://github.com/rockchip-linux/rknn-toolkit2')
        sys.exit(1)

    print(f'[2/3] 转换 RKNN: {onnx_path} -> {rknn_path}')
    print(f'    目标平台: {platform}')
    print(f'    INT8 量化: {quantize}')

    rknn = RKNN()

    # 配置
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=platform,
    )

    # 加载 ONNX
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f'    [ERROR] 加载 ONNX 失败: {ret}')
        sys.exit(1)

    # 构建（量化或浮点）
    ret = rknn.build(do_quantization=quantize, dataset=dataset_path)
    if ret != 0:
        print(f'    [ERROR] 构建 RKNN 失败: {ret}')
        sys.exit(1)

    # 导出
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f'    [ERROR] 导出 RKNN 失败: {ret}')
        sys.exit(1)

    rknn.release()
    print(f'    RKNN 转换完成: {rknn_path}')


def main():
    parser = argparse.ArgumentParser(description='YOLOv5 PT -> ONNX -> RKNN')
    parser.add_argument('--pt', type=str, default='weights/yolov5s.pt',
                        help='输入 PyTorch 模型路径')
    parser.add_argument('--output', type=str, default='weights/yolov5s.rknn',
                        help='输出 RKNN 模型路径')
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图像尺寸')
    parser.add_argument('--platform', type=str, default='rk3588',
                        choices=['rk3588', 'rk3566', 'rk3568'],
                        help='目标平台')
    parser.add_argument('--no-quantize', action='store_true',
                        help='不做 INT8 量化（使用 FP16）')
    parser.add_argument('--dataset', type=str, default=None,
                        help='量化校准数据集路径（文本文件，每行一个图像路径）')
    args = parser.parse_args()

    onnx_path = args.pt.replace('.pt', '.onnx')

    print('=' * 60)
    print('YOLOv5 模型转换: PT -> ONNX -> RKNN')
    print('=' * 60)

    # Step 1: PT -> ONNX
    export_onnx(args.pt, onnx_path, args.img_size)

    # Step 2: ONNX -> RKNN
    convert_rknn(
        onnx_path, args.output,
        platform=args.platform,
        img_size=args.img_size,
        quantize=not args.no_quantize,
        dataset_path=args.dataset,
    )

    print()
    print(f'[3/3] 完成！部署步骤:')
    print(f'  1. 将 {args.output} 拷贝到 RK3588 设备')
    print(f'  2. 确认 rknn-toolkit-lite2 已安装')
    print(f'  3. 修改 config/yolov5s.yaml 中 inference_backend: "rknn"')
    print(f'     并设置 rknn_model: "{args.output}"')
    print('=' * 60)


if __name__ == '__main__':
    main()
