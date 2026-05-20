#!/usr/bin/env python3
"""
模型转换脚本: .pt -> .onnx (-> .rknn 可选)

支持导出为 ONNX（部署到 ONNX Runtime CPU 推理）
和 RKNN（部署到 RK3588 NPU 推理）两种格式。

用法:
    # 仅导出 ONNX（在任何平台上都可以运行）
    python scripts/export_model.py --pt newckpt/best.pt --format onnx

    # 导出 ONNX + RKNN（需要 x86 环境 + rknn-toolkit2）
    python scripts/export_model.py --pt newckpt/best.pt --format rknn

    # 指定输出目录
    python scripts/export_model.py --pt newckpt/best.pt --output-dir weights/

依赖:
    - torch, torchvision (导出 ONNX)
    - onnx, onnxsim (ONNX 验证 + 简化，可选)
    - rknn-toolkit2 (ONNX -> RKNN，仅 x86，可选)
"""
import argparse
import os
import sys


def export_onnx(pt_path, onnx_path, img_size=640, simplify=True):
    """导出 ONNX 模型"""
    import torch

    print(f'[1/2] 导出 ONNX: {pt_path} -> {onnx_path}')

    ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
    model = ckpt.get('ema') or ckpt.get('model')
    model = model.float().eval()

    # 打印类别信息
    if hasattr(model, 'names'):
        names = model.names
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names.keys())]
        print(f'    模型类别 ({len(names)}): {names}')

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

    # 简化
    if simplify:
        try:
            import onnxsim
            onnx_model = onnx.load(onnx_path)
            model_sim, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(model_sim, onnx_path)
                print('    ONNX 模型已简化')
            else:
                print('    [WARN] ONNX 简化验证失败，保留原始模型')
        except ImportError:
            print('    [INFO] onnxsim 未安装，跳过简化 (pip install onnxsim)')

    return onnx_path


def convert_rknn(onnx_path, rknn_path, platform='rk3588', img_size=640,
                 quantize=True, dataset_path=None):
    """ONNX -> RKNN 转换"""
    try:
        from rknn.api import RKNN
    except ImportError:
        print('[ERROR] rknn-toolkit2 未安装。')
        print('  请在 x86 环境安装: pip install rknn-toolkit2')
        print('  参考: https://github.com/airockchip/rknn-toolkit2')
        sys.exit(1)

    print(f'[2/2] 转换 RKNN: {onnx_path} -> {rknn_path}')
    print(f'    目标平台: {platform}')
    print(f'    INT8 量化: {quantize}')

    rknn = RKNN()

    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=platform,
    )

    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f'    [ERROR] 加载 ONNX 失败: {ret}')
        sys.exit(1)

    ret = rknn.build(do_quantization=quantize, dataset=dataset_path)
    if ret != 0:
        print(f'    [ERROR] 构建 RKNN 失败: {ret}')
        sys.exit(1)

    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f'    [ERROR] 导出 RKNN 失败: {ret}')
        sys.exit(1)

    rknn.release()
    print(f'    RKNN 转换完成: {rknn_path}')


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv5 模型转换 (.pt -> .onnx / .rknn)')
    parser.add_argument('--pt', type=str, required=True,
                        help='输入 PyTorch 模型路径 (.pt)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认与输入文件同目录）')
    parser.add_argument('--format', type=str, default='onnx',
                        choices=['onnx', 'rknn', 'both'],
                        help='输出格式: onnx / rknn / both')
    parser.add_argument('--img-size', type=int, default=640,
                        help='模型输入尺寸')
    parser.add_argument('--platform', type=str, default='rk3588',
                        choices=['rk3588', 'rk3566', 'rk3568'],
                        help='RKNN 目标平台')
    parser.add_argument('--no-quantize', action='store_true',
                        help='RKNN 不做 INT8 量化')
    parser.add_argument('--no-simplify', action='store_true',
                        help='不简化 ONNX 模型')
    parser.add_argument('--dataset', type=str, default=None,
                        help='量化校准数据集（文本文件，每行一个图像路径）')
    args = parser.parse_args()

    if not os.path.isfile(args.pt):
        print(f'[ERROR] 模型文件不存在: {args.pt}')
        sys.exit(1)

    # 确定输出路径
    basename = os.path.splitext(os.path.basename(args.pt))[0]
    out_dir = args.output_dir or os.path.dirname(args.pt) or '.'
    os.makedirs(out_dir, exist_ok=True)

    onnx_path = os.path.join(out_dir, f'{basename}.onnx')
    rknn_path = os.path.join(out_dir, f'{basename}.rknn')

    print('=' * 60)
    print(f'YOLOv5 模型转换')
    print(f'  输入: {args.pt}')
    print(f'  格式: {args.format}')
    print(f'  尺寸: {args.img_size}x{args.img_size}')
    print('=' * 60)

    # ONNX 导出
    export_onnx(args.pt, onnx_path, args.img_size, simplify=not args.no_simplify)

    # RKNN 转换（可选）
    if args.format in ('rknn', 'both'):
        convert_rknn(
            onnx_path, rknn_path,
            platform=args.platform,
            img_size=args.img_size,
            quantize=not args.no_quantize,
            dataset_path=args.dataset,
        )

    print()
    print('=' * 60)
    print('转换完成！')
    print()
    print('部署步骤:')
    print(f'  1. 将 {onnx_path} 拷贝到 panel_detection/weights/ 目录')
    if args.format in ('rknn', 'both'):
        print(f'  2. 将 {rknn_path} 拷贝到 RK3588 设备的 weights/ 目录')
        print(f'  3. 修改 config/panel_detection.yaml:')
        print(f'       inference_backend: "rknn"')
        print(f'       rknn_model: "weights/{basename}.rknn"')
    else:
        print(f'  2. 修改 config/panel_detection.yaml:')
        print(f'       inference_backend: "onnx"')
        print(f'       onnx_model: "weights/{basename}.onnx"')
    print('=' * 60)


if __name__ == '__main__':
    main()
