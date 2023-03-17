import cv2
import numpy as np

# Load the image
image = cv2.imread("C:\Users\HP\Documents\kgfoss\Face-X\spiderman filter.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Apply adaptive thresholding to the blurred image
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

# Apply morphological operations to clean the image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Apply the spider filter
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, spider_filter = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)

# Display the results
cv2.imshow('Original', image)
cv2.imshow('Spider Filter', spider_filter)
cv2.waitKey(0)
cv2.destroyAllWindows()