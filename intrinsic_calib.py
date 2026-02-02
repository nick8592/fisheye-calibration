import cv2
import numpy as np
import os
import glob
import sys

# Hardcoded Parameters
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1536
BOARD_WIDTH = 18   # Internal corners width
BOARD_HEIGHT = 13  # Internal corners height
SQUARE_SIZE = 14   # mm

# Calibration settings
SUBPIX_REGION = 5
CALIB_NUMBER = 5 # Minimum number of valid frames required

def main():
    # Helper to find images. Defaults to ./data/*.jpg if no arg provided
    if len(sys.argv) > 1:
        img_mask = sys.argv[1]
    else:
        img_mask = './data/*.jpg'
    
    images = glob.glob(img_mask)
    if not images:
        print(f"No images found matching: {img_mask}")
        sys.exit(1)

    print(f"Found {len(images)} images. Processing...")

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(BOARD_WIDTH-1, BOARD_HEIGHT-1, 0)
    # Generate points in row-major order: (0,0), (1,0), (2,0)... then (0,1), (1,1)...
    # Shape: (N, 1, 3)
    objp = np.array([ [(j * SQUARE_SIZE, i * SQUARE_SIZE, 0.)]
                               for i in range(BOARD_HEIGHT) 
                               for j in range(BOARD_WIDTH) ], dtype=np.float32)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    valid_images = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (BOARD_WIDTH, BOARD_HEIGHT), 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

        if ret == True:
            objpoints.append(objp)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            corners2 = cv2.cornerSubPix(gray, corners, (SUBPIX_REGION, SUBPIX_REGION), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            valid_images += 1
            print(f"Corners found in {fname}")
        else:
            print(f"Corners NOT found in {fname}")

    if valid_images < CALIB_NUMBER:
        print(f"Not enough valid images for calibration. Found {valid_images}, need {CALIB_NUMBER}.")
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
    # Using constants from requirements:
    frame_size = (FRAME_WIDTH, FRAME_HEIGHT)

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
