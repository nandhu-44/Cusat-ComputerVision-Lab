import cv2
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────
#  Option A: Best looking result (recommended)
# ───────────────────────────────────────────────
def sift_match_beautiful(img1_path, img2_path, min_match_count=10, output_file=None):
    """
    SIFT + ratio test + homography + nice visualization
    """
    # 1. Read images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # query
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)  # train

    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both images could not be read")

    # 2. Initiate SIFT detector
    sift = cv2.SIFT_create(nfeatures=8000,   # increase if you have large images
                           contrastThreshold=0.03,
                           edgeThreshold=10,
                           sigma=1.6)

    # find the keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(f"Keypoints:  img1 = {len(kp1):4d}   img2 = {len(kp2):4d}")

    # 3. FLANN based matcher (faster for SIFT)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 4. Apply Lowe's ratio test
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Good matches after ratio test: {len(good)}")

    # 5. Visualize matches (two common styles)

    # ── Style A: Classic matching lines ───────────────────────
    draw_params = dict(
        matchColor=(0, 255, 0),      # draw matches in green
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_DEFAULT
    )

    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, good, None,
        **draw_params
    )

    # ── Style B: Only show "confident" matches with homography ──
    if len(good) >= min_match_count:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        print(f"Inliers after RANSAC: {sum(matchesMask)} / {len(good)}")

        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(0, 0, 255),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT
        )

        img_inliers = cv2.drawMatches(
            img1, kp1, img2, kp2, good, None,
            **draw_params
        )

        # Optional: draw outline of projected region
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)

        img2_with_box = cv2.polylines(
            cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR),
            [np.int32(dst)], True, (0, 255, 255), 3, cv2.LINE_AA
        )

    else:
        print("Not enough good matches → Homography skipped")
        img_inliers = None
        img2_with_box = None

    # ── Save results ───────────────────────────────────────────
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Image 1 (query)')

    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Image 2 (scene)')

    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'All good matches (ratio < 0.75) — {len(good)} matches')

    plt.tight_layout()
    plt.savefig('sift_matching_all_matches.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Optional second figure with inliers + projection box
    if img_inliers is not None:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_inliers, cv2.COLOR_BGR2RGB))
        plt.title(f'Inlier matches after RANSAC ({sum(matchesMask)})')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img2_with_box, cv2.COLOR_BGR2RGB))
        plt.title('Projected region from img1 → img2')

        plt.tight_layout()
        plt.savefig('sift_matching_inliers_and_projection.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Save text results
    if output_file:
        with open(output_file, 'w') as f:
            f.write(f"Keypoints: img1 = {len(kp1)}, img2 = {len(kp2)}\n")
            f.write(f"Good matches after ratio test: {len(good)}\n")
            if len(good) >= min_match_count:
                f.write(f"Inliers after RANSAC: {sum(matchesMask)} / {len(good)}\n")
                f.write("Homography matrix:\n")
                f.write(str(H) + "\n")
            else:
                f.write("Not enough good matches → Homography skipped\n")


# ───────────────────────────────────────────────
#  Usage example
# ───────────────────────────────────────────────
if __name__ == "__main__":

    # Change these paths
    path1 = "images/sift-image-1.jpeg"
    path2 = "images/sift-image-2.jpeg"

    try:
        sift_match_beautiful(path1, path2, min_match_count=12, output_file='sift_matching_results.txt')
        print("Results saved to 'sift_matching_results.txt' and images saved to PNG files.")
    except Exception as e:
        print("Error:", str(e))