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

### Custom calibration parameters:
```bash
python intrinsic_calib.py "Images/cam0/*.jpg" -bw 18 -bh 13 -s 14
```

## Parameters

Calibration parameters can now be passed as arguments:

| Argument | Flag | Default | Description |
|----------|------|---------|-------------|
| `input_path` | (pos) | `./data/*.jpg` | Input images path glob pattern |
| `--frame-width` | `-fw` | `1920` | Frame Width |
| `--frame-height` | `-fh` | `1536` | Frame Height |
| `--board-width` | `-bw` | `18` | Board Width (internal corners) |
| `--board-height` | `-bh` | `13` | Board Height (internal corners) |
| `--square-size` | `-s` | `14` | Square Size in mm |
| `--subpix-region` | `-sub` | `5` | Subpixel Region size |
| `--calib-number` | `-n` | `5` | Minimum number of valid frames |

## Output

The script outputs:
- **RMS Error**: Root Mean Square error returned by `cv2.fisheye.calibrate`.
- **Reprojection Error**: Mean L2 norm of the reprojection error.
- **Camera Matrix (K)**: Saved to `camera_K.npy`.
- **Distortion Coefficients (D)**: Saved to `camera_D.npy`.