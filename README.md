# SIFT Image Matching with Homography

This project implements SIFT (Scale-Invariant Feature Transform) based image matching between two images, using Lowe's ratio test and homography estimation with RANSAC to filter out outliers and visualize only the inlier matches.

## Requirements

- Python 3.x
- OpenCV (with contrib modules for SIFT)
- NumPy
- Matplotlib

## Installation

The required packages are already installed in the virtual environment.

## Usage

1. Place two images named `sift-image-1.jpeg` and `sift-image-2.jpeg` in the `images/` directory.

2. Run the script:

```bash
python sift_matching.py
```

The script will:

- Load the two images and convert to grayscale
- Detect SIFT keypoints and compute descriptors with optimized parameters
- Use FLANN-based matcher with k-NN (k=2) for efficient matching
- Apply Lowe's ratio test (0.75 threshold) to filter good matches
- If enough good matches (>12), compute homography using RANSAC
- Save two output images:
  - `sift_matching_all_matches.png`: Shows all good matches
  - `sift_matching_inliers_and_projection.png`: Shows inlier matches and projected object boundary
- Save text results to `sift_matching_results.txt` including keypoint counts, match statistics, and homography matrix

## Output Files

- `sift_matching_results.txt`: Text file containing detailed results
- `sift_matching_all_matches.png`: Visualization of all good matches
- `sift_matching_inliers_and_projection.png`: Inlier matches with object projection (if homography successful)

## How it works

1. **Feature Detection**: Uses SIFT to detect keypoints and compute 128-dimensional descriptors.

2. **Feature Matching**: Employs FLANN (Fast Library for Approximate Nearest Neighbors) for efficient matching, followed by Lowe's ratio test to eliminate ambiguous matches.

3. **Outlier Removal**: Uses RANSAC algorithm within `cv2.findHomography` to robustly estimate the homography matrix and identify inlier matches.

4. **Object Detection**: Transforms the corners of the first image to the second image using the homography matrix and draws a bounding polygon.

5. **Visualization**: Displays the matched images with only inlier matches shown in green, and the detected object outlined in white.

## Improvements over Basic Matching

- **Ratio Test**: Filters matches based on distance ratio to reduce false positives.
- **Homography Estimation**: Uses RANSAC to find the geometric transformation between images.
- **Inlier Visualization**: Only shows matches that are consistent with the estimated homography.
- **Object Localization**: Draws the bounding box of the detected object in the scene.

## Notes

- SIFT is patented, but available in OpenCV for educational purposes.
- Requires at least 10 good matches to attempt homography estimation.
- If the images don't contain the same object/scene, it will show "Not enough matches" message.
- The ratio threshold (0.7) and RANSAC parameters can be tuned for different scenarios.
