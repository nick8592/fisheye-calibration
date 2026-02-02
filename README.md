# Fisheye Calibration

This project performs intrinsic camera calibration for fisheye lenses using OpenCV.

## Requirements

- Python 3.8
- Dependencies listed in `requirements.txt`

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the calibration script `intrinsic_calib.py`. By default, it looks for images in `./data/*.jpg`. You can specify a different path using a glob pattern.

### Default usage:
```bash
python intrinsic_calib.py
```

### Specifying image path:
```bash
python intrinsic_calib.py "Images/cam0/*.jpg"
```

## Parameters

The calibration parameters are hardcoded in `intrinsic_calib.py`:
- **Frame Size**: 1920x1536
- **Checkerboard**: 18x13 (internal corners)
- **Square Size**: 14mm

## Output

The script outputs:
- **RMS Error**: Root Mean Square error returned by `cv2.fisheye.calibrate`.
- **Reprojection Error**: Mean L2 norm of the reprojection error.
- **Camera Matrix (K)**: Saved to `camera_K.npy`.
- **Distortion Coefficients (D)**: Saved to `camera_D.npy`.