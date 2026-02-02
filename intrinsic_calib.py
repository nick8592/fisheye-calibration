import cv2
import numpy as np
import os
import glob
import sys

import argparse

def main():
    parser = argparse.ArgumentParser(description='Fisheye Camera Calibration')
    
    # Input path (positional optional)
    parser.add_argument('input_path', nargs='?', default='./data/*.jpg', help='Input images path glob pattern (default: ./data/*.jpg)')
    
    # Calibration Parameters
    parser.add_argument('-fw', '--frame-width', type=int, default=1920, help='Frame Width (default: 1920)')
    parser.add_argument('-fh', '--frame-height', type=int, default=1536, help='Frame Height (default: 1536)')
    parser.add_argument('-bw', '--board-width', type=int, default=18, help='Board Width (internal corners) (default: 18)')
    parser.add_argument('-bh', '--board-height', type=int, default=13, help='Board Height (internal corners) (default: 13)')
    parser.add_argument('-s', '--square-size', type=float, default=14, help='Square Size in mm (default: 14)')
    parser.add_argument('-sub', '--subpix-region', type=int, default=5, help='Subpixel Region size (default: 5)')
    parser.add_argument('-n', '--calib-number', type=int, default=5, help='Minimum number of valid frames (default: 5)')

    args = parser.parse_args()
    
    img_mask = args.input_path
    
    images = glob.glob(img_mask)
    if not images:
        print(f"No images found matching: {img_mask}")
        sys.exit(1)

    print(f"Found {len(images)} images. Processing...")

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(BOARD_WIDTH-1, BOARD_HEIGHT-1, 0)
    # Generate points in row-major order: (0,0), (1,0), (2,0)... then (0,1), (1,1)...
    # Shape: (N, 1, 3)
    objp = np.array([ [(j * args.square_size, i * args.square_size, 0.)]
                               for i in range(args.board_height) 
                               for j in range(args.board_width) ], dtype=np.float32)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    valid_images = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (args.board_width, args.board_height), 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

        if ret == True:
            objpoints.append(objp)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            corners2 = cv2.cornerSubPix(gray, corners, (args.subpix_region, args.subpix_region), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            valid_images += 1
            print(f"Corners found in {fname}")
        else:
            print(f"Corners NOT found in {fname}")

    if valid_images < args.calib_number:
        print(f"Not enough valid images for calibration. Found {valid_images}, need {args.calib_number}.")
        sys.exit(1)

    print("Calibrating...")
    
    # Fisheye calibration
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(valid_images)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(valid_images)]
    
    # Fisheye calibration flags
    # CALIB_FIX_SKEW: Skew coefficient (alpha) is set to zero and stays zero
    # CALIB_RECOMPUTE_EXTRINSIC: Extrinsic will be recomputed after each iteration of intrinsic optimization
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    
    # Input frame size (width, height)
    # Assuming all images are same size as the first one found useful, or using constants.
    frame_size = (args.frame_width, args.frame_height)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, 
        imgpoints, 
        frame_size, 
        K, 
        D, 
        rvecs, 
        tvecs, 
        calibration_flags, 
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    print("Calibration successfully finished!")
    print(f"RMS: {rms}")
    
    # Calculate reprojection error
    # Metric: mean( norm(error_vector) / num_points ) per image

    reproj_errs = []
    for i in range(len(objpoints)):
        corners_reproj, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(corners_reproj, imgpoints[i], cv2.NORM_L2) / len(corners_reproj)
        reproj_errs.append(err)
    
    print(f"Reprojection Error: {np.mean(reproj_errs)}")

    print(f"Camera Matrix (K):\n{K}")
    print(f"Distortion Coeffs (D):\n{D}")

    # Save results
    np.save('camera_K.npy', K)
    np.save('camera_D.npy', D)
    print("Saved K and D to camera_K.npy and camera_D.npy")

if __name__ == '__main__':
    main()
