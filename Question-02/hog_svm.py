"Implement a pedestrian detection system using HOG features and SVM classifier and evaluate its performance."

import os
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import hog as sk_hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump


def read_image_grayscale(path, size=(64, 128)):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size[0], size[1]))
    return img


def extract_hog_features(image):
    features = sk_hog(image,
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      visualize=False,
                      feature_vector=True)
    return features


def load_dataset(pos_dir, neg_dir, max_neg=None):
    X = []
    y = []

    pos_paths = list(Path(pos_dir).glob('*.*'))
    neg_paths = list(Path(neg_dir).glob('*.*'))

    for p in pos_paths:
        img = read_image_grayscale(p)
        if img is None:
            continue
        X.append(extract_hog_features(img))
        y.append(1)

    if max_neg:
        neg_paths = neg_paths[:max_neg]

    for n in neg_paths:
        img = read_image_grayscale(n)
        if img is None:
            continue
        X.append(extract_hog_features(img))
        y.append(0)

    return np.array(X), np.array(y)


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / 'data' / 'INRIA-Dataset'

    # Prefer canonical Train/Test layout; fall back to common alternate folders
    train_pos = data_dir / 'Train' / 'pos'
    train_neg = data_dir / 'Train' / 'neg'
    test_pos = data_dir / 'Test' / 'pos'
    test_neg = data_dir / 'Test' / 'neg'

    for p in (train_pos, train_neg, test_pos, test_neg):
        if not p.exists():
            print(f'Missing expected folder: {p}')

    print('Loading training data...')
    X_train, y_train = load_dataset(train_pos, train_neg, max_neg=5000)
    print(f'Training samples: {len(y_train)} (pos: {y_train.sum()}, neg: {len(y_train)-y_train.sum()})')

    print('Training LinearSVC on HOG features...')
    clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))
    clf.fit(X_train, y_train)

    print('Loading test data...')
    X_test, y_test = load_dataset(test_pos, test_neg, max_neg=2000)
    print(f'Test samples: {len(y_test)} (pos: {y_test.sum()}, neg: {len(y_test)-y_test.sum()})')

    print('Evaluating...')
    y_pred = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification report:\n', classification_report(y_test, y_pred, digits=4))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'hog_svm_model.joblib'
    dump(clf, model_path)
    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()