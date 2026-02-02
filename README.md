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
python intrinsic_calib.py -i "Images/cam0/*.jpg"
```

### Custom calibration parameters:
```bash
python intrinsic_calib.py -i "Images/cam0/*.jpg" -bw 18 -bh 13 -s 14
```

## Parameters

Calibration parameters can now be passed as arguments:

| Argument | Flag | Default | Description |
|----------|------|---------|-------------|
| `--input-path` | `-i` | `./data/*.jpg` | Input images path glob pattern |
| `--frame-width` | `-fw` | `1920` | Frame Width |
| `--frame-height` | `-fh` | `1536` | Frame Height |
| `--board-width` | `-bw` | `18` | Board Width (internal corners) |
| `--board-height` | `-bh` | `13` | Board Height (internal corners) |
| `--square-size` | `-s` | `14` | Square Size in mm |
| `--subpix-region` | `-sub` | `5` | Subpixel Region size |
| `--calib-number` | `-n` | `5` | Minimum number of valid frames |
| `--output-dir` | `-o` | `results` | Directory to save calibration results (npy files) |
| `--save-undistorted` | | `False` | Flag to save undistorted images to `undistorted_images/` |
| `--undistort-mode` | | `balance` | `original` (use K) or `balance` (use estimated K) |
| `--balance` | | `0.5` | Balance factor for `balance` mode (0.0=crop, 1.0=full view) |

## Undistortion Modes

You can control how the undistorted images are generated using `--undistort-mode` and `--balance` (only applies if `--save-undistorted` is used).

### 1. Original Mode (`--undistort-mode original`)
Uses the calibrated Camera Matrix (`K`).
- **Effect**: "Safety mode". It cuts out all curved edges to ensure every pixel in the output is strictly valid.
- **Result**: Significant crop. You see the center of the image, but lose the wide-angle edges.

### 2. Balance Mode (`--undistort-mode balance`)
Uses a new estimated Camera Matrix (`K_new`) via `cv2.fisheye.estimateNewCameraMatrixForUndistortRectify`.
- **`--balance 0.0`**: Equivalent to using `K`. Max crop, no black borders.
- **`--balance 1.0`**: Max Field of View. Preserves all source pixels but introduces black wavy borders at the edges.
- **`--balance 0.5`**: A mix between keeping FOV and cropping.

## Output

The script outputs:
- **RMS Error**: Root Mean Square error returned by `cv2.fisheye.calibrate`.
- **Reprojection Error**: Mean L2 norm of the reprojection error.
- **Calibration Files**: Saved to `output_dir` (default: `results/`).
  - `camera_K.npy`
  - `camera_D.npy`
- **Undistorted Images**: Saved to `results/undistorted_images/` directory if `--save-undistorted` is used.