#!/usr/bin/env python3
"""
ChArUco 标定板相机标定工具

使用 Orbbec Gemini 336 拍摄 ChArUco 标定板，标定彩色相机内参，
并与出厂内参对比。

标定板规格（默认）:
    - ChArUco 5x7（5列7行角点网格，即 6x8 方格）
    - 方格边长 30mm，ArUco 标记边长 22mm
    - 字典 DICT_4X4_50

用法:
    python tools/calibrate_charuco.py                    # 实时采集 + 标定
    python tools/calibrate_charuco.py --images cal_imgs/ # 从已有图片标定
    python tools/calibrate_charuco.py --num 20           # 采集 20 张后标定

操作:
    空格/s  采集当前帧
    c       开始标定（采集够后）
    q/ESC   退出
"""
import argparse
import glob
import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def create_charuco_board(cols=5, rows=7, square_size=0.030, marker_size=0.022):
    """创建 ChArUco 标定板和检测器"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((cols, rows), square_size, marker_size, dictionary)
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    return board, charuco_detector


def detect_charuco(image, board, charuco_detector):
    """检测 ChArUco 角点，返回 (charuco_corners, charuco_ids, gray)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if charuco_ids is None or len(charuco_ids) < 6:
        return None, None, gray

    return charuco_corners, charuco_ids, gray


def draw_charuco(image, charuco_corners, charuco_ids):
    """在图像上绘制检测到的角点"""
    vis = image.copy()
    if charuco_corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids,
                                              cornerColor=(0, 255, 0))
        cv2.putText(vis, f'Corners: {len(charuco_ids)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(vis, 'No board detected', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis


def calibrate(all_corners, all_ids, board, image_size):
    """执行标定"""
    # OpenCV 4.13+: 用标准 calibrateCamera，通过 board 获取物体点
    obj_points = []
    img_points = []
    for corners, ids in zip(all_corners, all_ids):
        # 获取每张图中检测到的角点对应的 3D 物体坐标
        obj_pts = board.getChessboardCorners()  # 所有角点的 3D 坐标
        # 只取检测到的 id 对应的点
        matched_obj = np.array([obj_pts[i[0]] for i in ids], dtype=np.float32)
        obj_points.append(matched_obj)
        img_points.append(corners.reshape(-1, 1, 2).astype(np.float32))

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    return ret, camera_matrix, dist_coeffs


def print_comparison(cal_matrix, cal_dist, factory_intrin):
    """打印标定结果与出厂内参的对比"""
    cal_fx = cal_matrix[0, 0]
    cal_fy = cal_matrix[1, 1]
    cal_cx = cal_matrix[0, 2]
    cal_cy = cal_matrix[1, 2]

    fac_fx = factory_intrin.fx
    fac_fy = factory_intrin.fy
    fac_cx = factory_intrin.cx
    fac_cy = factory_intrin.cy
    fac_coeffs = factory_intrin.coeffs

    print()
    print('=' * 60)
    print('  标定结果 vs 出厂内参 对比')
    print('=' * 60)
    print(f'{"参数":<12} {"标定值":>12} {"出厂值":>12} {"差异":>10} {"差异%":>8}')
    print('-' * 60)

    for name, cal, fac in [
        ('fx', cal_fx, fac_fx),
        ('fy', cal_fy, fac_fy),
        ('cx', cal_cx, fac_cx),
        ('cy', cal_cy, fac_cy),
    ]:
        diff = cal - fac
        pct = diff / fac * 100 if fac != 0 else 0
        print(f'{name:<12} {cal:>12.4f} {fac:>12.4f} {diff:>+10.4f} {pct:>+7.2f}%')

    print()
    print('畸变系数 [k1, k2, p1, p2, k3]:')
    cal_d = cal_dist.flatten()
    for i, name in enumerate(['k1', 'k2', 'p1', 'p2', 'k3']):
        cal_v = cal_d[i] if i < len(cal_d) else 0
        fac_v = fac_coeffs[i] if i < len(fac_coeffs) else 0
        print(f'  {name}: 标定={cal_v:>12.6f}  出厂={fac_v:>12.6f}  差={cal_v - fac_v:>+12.6f}')

    if len(cal_d) > 5:
        print(f'  额外系数: {cal_d[5:].tolist()}')

    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(description='ChArUco 标定板相机标定')
    parser.add_argument('--images', default=None, help='从已有图片目录标定')
    parser.add_argument('--num', type=int, default=15, help='采集张数（默认 15）')
    parser.add_argument('--cols', type=int, default=5, help='标定板列数')
    parser.add_argument('--rows', type=int, default=7, help='标定板行数')
    parser.add_argument('--square', type=float, default=0.030, help='方格边长（米）')
    parser.add_argument('--marker', type=float, default=0.022, help='ArUco 标记边长（米）')
    parser.add_argument('--save-dir', default='calibration', help='保存目录')
    args = parser.parse_args()

    board, detector = create_charuco_board(args.cols, args.rows, args.square, args.marker)
    print(f'[INFO] 标定板: ChArUco {args.cols}x{args.rows}, 方格={args.square*1000:.0f}mm, 标记={args.marker*1000:.0f}mm')

    # ── 从图片标定 ────────────────────────────────────────────────
    if args.images:
        image_files = sorted(glob.glob(os.path.join(args.images, '*.jpg')) +
                             glob.glob(os.path.join(args.images, '*.png')))
        if not image_files:
            print(f'[ERROR] {args.images} 中没有图片')
            return

        print(f'[INFO] 从 {len(image_files)} 张图片标定')
        all_corners, all_ids = [], []
        image_size = None

        for f in image_files:
            img = cv2.imread(f)
            if img is None:
                continue
            if image_size is None:
                image_size = (img.shape[1], img.shape[0])
            corners, ids, _ = detect_charuco(img, board, detector)
            if corners is not None:
                all_corners.append(corners)
                all_ids.append(ids)
                print(f'  {os.path.basename(f)}: {len(ids)} 角点')
            else:
                print(f'  {os.path.basename(f)}: 未检测到')

        if len(all_corners) < 5:
            print(f'[ERROR] 有效图片太少 ({len(all_corners)})，至少需要 5 张')
            return

        ret, cam_matrix, dist_coeffs = calibrate(all_corners, all_ids, board, image_size)
        print(f'\n[INFO] 标定完成，重投影误差: {ret:.4f} 像素')
        # 无出厂内参可对比
        print(f'\n标定内参:')
        print(f'  fx={cam_matrix[0,0]:.4f} fy={cam_matrix[1,1]:.4f}')
        print(f'  cx={cam_matrix[0,2]:.4f} cy={cam_matrix[1,2]:.4f}')
        print(f'  dist={dist_coeffs.flatten()[:5].tolist()}')
        return

    # ── 实时采集 + 标定 ───────────────────────────────────────────
    from camera.orbbec import OrbbecBackend
    import app_config

    app_config.load_config()
    cam = OrbbecBackend()
    cam_cfg = app_config.config.get('camera', {})
    cw = cam_cfg.get('color_width', 640)
    ch = cam_cfg.get('color_height', 480)
    cam.initialize(cw, ch, cw, ch, cam_cfg.get('fps', 30))

    factory_intrin = cam._color_intrinsics
    image_size = (cw, ch)

    os.makedirs(args.save_dir, exist_ok=True)

    all_corners, all_ids = [], []
    collected = 0
    last_collect_time = 0

    print(f'[INFO] 目标采集 {args.num} 张，保存到 {args.save_dir}/')
    print('[INFO] 操作: 空格/s=采集  c=标定  q=退出')
    print('[INFO] 建议：多角度、多距离拍摄，覆盖画面各区域')
    print()

    while True:
        result = cam.get_aligned_frames()
        if result[0] is None:
            time.sleep(0.05)
            continue

        _, _, color_image, _ = result
        corners, ids, gray = detect_charuco(color_image, board, detector)
        vis = draw_charuco(color_image, corners, ids)

        # 显示采集进度
        cv2.putText(vis, f'Collected: {collected}/{args.num}', (10, ch - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if collected >= args.num:
            cv2.putText(vis, 'Press C to calibrate', (10, ch - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.namedWindow('calibration', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('calibration', vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break

        # 采集
        if key in (ord(' '), ord('s')):
            now = time.time()
            if corners is not None and now - last_collect_time > 0.5:
                all_corners.append(corners)
                all_ids.append(ids)
                collected += 1
                last_collect_time = now

                # 保存图片
                fname = os.path.join(args.save_dir, f'cal_{collected:03d}.jpg')
                cv2.imwrite(fname, color_image)
                print(f'  [{collected}/{args.num}] 采集成功，{len(ids)} 角点 -> {fname}')
            elif corners is None:
                print('  未检测到标定板，请调整角度')

        # 标定
        if key == ord('c') and collected >= 5:
            print(f'\n[INFO] 开始标定（{collected} 张图片）...')
            ret, cam_matrix, dist_coeffs = calibrate(all_corners, all_ids, board, image_size)
            print(f'[INFO] 重投影误差: {ret:.4f} 像素')

            print_comparison(cam_matrix, dist_coeffs, factory_intrin)

            # 保存标定结果
            result_file = os.path.join(args.save_dir, 'calibration_result.npz')
            np.savez(result_file,
                     camera_matrix=cam_matrix,
                     dist_coeffs=dist_coeffs,
                     reprojection_error=ret,
                     image_size=np.array(image_size))
            print(f'\n[INFO] 标定结果已保存: {result_file}')

            # 保存为 YAML（方便其他工具读取）
            yaml_file = os.path.join(args.save_dir, 'calibration_result.yaml')
            fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_WRITE)
            fs.write('camera_matrix', cam_matrix)
            fs.write('dist_coeffs', dist_coeffs)
            fs.write('reprojection_error', ret)
            fs.write('image_width', image_size[0])
            fs.write('image_height', image_size[1])
            fs.release()
            print(f'[INFO] YAML 已保存: {yaml_file}')
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
