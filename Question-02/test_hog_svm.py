#!/usr/bin/env python3
"""Load the trained HOG+SVM model and run sliding-window detection on an image.

Usage:
    python Question-02/test_hog_svm.py --image ../images/pedestrian-detection.jpg
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog as sk_hog
from imutils.object_detection import non_max_suppression
from joblib import load


def pyramid(image, scale=1.25, minSize=(64, 128)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        if w < minSize[0] or h < minSize[1]:
            break
        image = cv2.resize(image, (w, h))
        yield image


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0] - windowSize[1] + 1, stepSize):
        for x in range(0, image.shape[1] - windowSize[0] + 1, stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def extract_hog_features(image):
    return sk_hog(image,
                  orientations=9,
                  pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2),
                  block_norm='L2-Hys',
                  visualize=False,
                  feature_vector=True)


def detect(image_path, model_path, threshold=0.0, step_size=8, visualize=False):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')

    clf = load(model_path)

    orig = cv2.imread(str(image_path))
    if orig is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    image = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    winW, winH = 64, 128
    rects = []
    scores = []

    for resized in pyramid(image, scale=1.25, minSize=(winW, winH)):
        scale_x = orig.shape[1] / float(resized.shape[1])
        scale_y = orig.shape[0] / float(resized.shape[0])

        for (x, y, window) in sliding_window(resized, step_size, (winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            feat = extract_hog_features(window)
            feat = feat.reshape(1, -1)
            try:
                score = clf.decision_function(feat)
            except Exception:
                score = clf.predict_proba(feat)[:, 1]
            score = float(np.ravel(score)[0])
            if score >= threshold:
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int((x + winW) * scale_x)
                y2 = int((y + winH) * scale_y)
                rects.append([x1, y1, x2, y2])
                scores.append(score)

    rects = np.array(rects)
    scores = np.array(scores)

    if len(rects) > 0:
        pick = non_max_suppression(rects, probs=scores, overlapThresh=0.3)
    else:
        pick = []

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(orig, (xA, yA), (xB, yB), (0, 255, 0), 2)

    out_path = Path.cwd() / 'output_detected.jpg'
    cv2.imwrite(str(out_path), orig)
    print(f'Detections: {len(pick)}. Output saved to: {out_path}')

    if visualize:
        cv2.imshow('Detections', orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', default='../images/pedestrian-detection.jpg')
    parser.add_argument('--model', '-m', default='../models/hog_svm_model.joblib')
    parser.add_argument('--threshold', '-t', type=float, default=0.0)
    parser.add_argument('--step', type=int, default=8)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    detect(args.image, args.model, threshold=args.threshold, step_size=args.step, visualize=args.visualize)


if __name__ == '__main__':
    main()
