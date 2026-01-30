"Implement a pedestrian detection system using HOG features and SVM classifier and evaluate its performance."

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression  # pip install imutils

# Initialize HOG descriptor with pre-trained SVM for pedestrians
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load and preprocess the image (replace 'input_image.jpg' with your file)
image = cv2.imread('../images/pedestrian-detection.jpg')
image = cv2.resize(image, (min(800, image.shape[1]), min(800, image.shape[0])))  # Resize for efficiency
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect pedestrians (multi-scale sliding window)
rects, weights = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

# Apply non-maximum suppression to remove overlapping boxes
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
picked = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# Draw bounding boxes on the image
for (xA, yA, xB, yB) in picked:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

# Save and display the result
cv2.imwrite('output_detected.jpg', image)
cv2.imshow('Pedestrian Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()