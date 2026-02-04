import cv2
import numpy as np
import os
import glob
import sys

import argparse
import datetime

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        self.terminal.flush()
        self.log.flush()

def main():
    parser = argparse.ArgumentParser(description='Fisheye Camera Calibration')
    
    # Input path (named argument)
    parser.add_argument('-i', '--input-path', type=str, default='./data/*.jpg', help='Input images path glob pattern (default: ./data/*.jpg)')
    
    # Calibration Parameters
    parser.add_argument('-fw', '--frame-width', type=int, default=1920, help='Frame Width (default: 1920)')
    parser.add_argument('-fh', '--frame-height', type=int, default=1536, help='Frame Height (default: 1536)')
    parser.add_argument('-bw', '--board-width', type=int, default=18, help='Board Width (internal corners) (default: 18)')
    parser.add_argument('-bh', '--board-height', type=int, default=13, help='Board Height (internal corners) (default: 13)')
    parser.add_argument('-s', '--square-size', type=float, default=14, help='Square Size in mm (default: 14)')
    parser.add_argument('-sub', '--subpix-region', type=int, default=5, help='Subpixel Region size (default: 5)')
    parser.add_argument('-n', '--calib-number', type=int, default=5, help='Minimum number of valid frames (default: 5)')
    parser.add_argument('-o', '--output-dir', type=str, default='results', help='Directory to save calibration results (npy files). Default: results')
    parser.add_argument('--save-undistorted', action='store_true', help='Save undistorted images to "undistorted_images" directory.')
    parser.add_argument('--undistort-mode', type=str, default='balance', choices=['original', 'balance'], help='Undistortion mode: "original" uses calibrated K, "balance" estimates new K. Default: balance')
    parser.add_argument('--balance', type=float, default=0.5, help='Balance factor for "balance" mode (0.0=max crop, 1.0=full view). Default: 0.5')

    args = parser.parse_args()

    # Ensure output directory exists before logging
    save_dir = args.output_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"calibration_{timestamp}.log")
    print(f"Logging to: {log_file}")
    
    # Redirect stdout and stderr to log file
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    print("="*40)
    print("Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("="*40)
    
    img_mask = args.input_path
    
    images = glob.glob(img_mask)
    if not images:
        print(f"No images found matching: {img_mask}")
        sys.exit(1)

    print(f"\n[1/4] Processing Images ({len(images)} files)...")

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(BOARD_WIDTH-1, BOARD_HEIGHT-1, 0)
    # Generate points in row-major order: (0,0), (1,0), (2,0)... then (0,1), (1,1)...
    # Shape: (N, 1, 3)
    objp = np.array([ [(j * args.square_size, i * args.square_size, 0.)]
                               for i in range(args.board_height) 
                               for j in range(args.board_width) ], dtype=np.float32)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    valid_filenames = []

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
            valid_filenames.append(fname)
            
            valid_images += 1
            print(f"[OK]   {fname}")
        else:
            print(f"[FAIL] {fname}")

    if valid_images < args.calib_number:
        print(f"\nError: Not enough valid images for calibration. Found {valid_images}, need {args.calib_number}.")
        sys.exit(1)

    print(f"\n[2/4] Calibrating with {valid_images} valid images...")
    
    # Fisheye calibration
    w = args.frame_width
    h = args.frame_height
    f_guess = w * 0.4  # Rough estimation of focal length
    
    K = np.array([
        [f_guess, 0, w/2],
        [0, f_guess, h/2],
        [0, 0, 1]
    ], dtype=np.float64)
    
    D = np.zeros((4, 1), dtype=np.float64)

    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(valid_images)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(valid_images)]
    
    # Fisheye calibration flags
    # CALIB_FIX_SKEW: Skew coefficient (alpha) is set to zero and stays zero
    # CALIB_RECOMPUTE_EXTRINSIC: Extrinsic will be recomputed after each iteration of intrinsic optimization
    # CALIB_USE_INTRINSIC_GUESS: Use the provided K matrix as an initial guess
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    
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

    print("\n[3/4] Calibration Results")
    print(f"RMS Error:          {rms}")

    # Calculate reprojection error
    # Metric: mean( norm(error_vector) / num_points ) per image
    reproj_errs = []
    
    # Store errors with filenames for sorting
    error_list = []

    for i in range(len(objpoints)):
        corners_reproj, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(corners_reproj, imgpoints[i], cv2.NORM_L2) / len(corners_reproj)
        reproj_errs.append(err)
        error_list.append((valid_filenames[i], err))
    
    mean_error = np.mean(reproj_errs)
    print(f"Reprojection Error: {mean_error}")
    
    print("\nPer-image Reprojection Errors (descending):")
    # Sort by error descending
    error_list.sort(key=lambda x: x[1], reverse=True)
    
    for fname, err in error_list:
        print(f"  {err:.5f} - {fname}")

    print("\nCamera Matrix (K):")
    print(K)
    print("\nDistortion Coeffs (D):")
    print(D)

    print("\n[4/4] Saving Results")
    # Save calibration results (.npy)
    save_dir = args.output_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'camera_K.npy'), K)
    np.save(os.path.join(save_dir, 'camera_D.npy'), D)
    print(f"Saved K and D to {save_dir}/camera_K.npy and {save_dir}/camera_D.npy")

    # Save undistorted images if requested
    if args.save_undistorted:
        undist_dir = f'{args.output_dir}/undistorted_images'
        if not os.path.exists(undist_dir):
            os.makedirs(undist_dir)
            
        print(f"Saving undistorted images to {undist_dir}...")
        
        # Determine New Camera Matrix
        if args.undistort_mode == 'original':
            # Use original calibrated matrix K (may result in heavy cropping)
            K_new = K
            print("Using original K matrix for undistortion.")
        else:
            # Estimate new matrix to control field of view vs valid pixels
            # balance=0.0 (max crop, no black borders), balance=1.0 (min crop, full FOV)
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, frame_size, np.eye(3), balance=args.balance)
            print(f"Using new estimated K matrix with balance={args.balance}")

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, frame_size, cv2.CV_16SC2)
        
        for fname in images:
            img = cv2.imread(fname)
            if img is None: continue
            
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            base_name = os.path.basename(fname)
            save_path = os.path.join(undist_dir, f"{base_name}")
            cv2.imwrite(save_path, undistorted_img)
            
        print("Undistorted images saved.")

if __name__ == '__main__':
    main()
