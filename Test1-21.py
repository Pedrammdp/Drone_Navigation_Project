import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('Pics/test1-2.png')
cv.imshow('image', img)

# Convert to HSV color space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('hsv', hsv)

# Threshold the image based on the gate color
lower_color = np.array([10, 100, 20])  # replace with the lower HSV values of the gate color
upper_color = np.array([25, 255, 255])  # replace with the upper HSV values of the gate color
mask = cv.inRange(hsv, lower_color, upper_color)

# Apply Gaussian blur to the image 
blurred = cv.GaussianBlur(mask, (5, 5), 0)
cv.imshow('blurred', blurred)

# Apply morphological operations to fill gaps in the gate contour
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
closed = cv.morphologyEx(blurred, cv.MORPH_CLOSE, kernel)
eroded = cv.erode(closed, kernel, iterations=2)
dilated = cv.dilate(eroded, kernel, iterations=2)
cv.imshow('Dilated', dilated)

# Apply Canny edge detection to detect the edges of the gate
edges = cv.Canny(dilated, 250, 300)
cv.imshow('edges', edges)

# Find contours in the edge image
contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours, -1, (0,255,0), 4)

# Drawing circle at the center
cnt = contours[0]
M = cv.moments(cnt)
print(M)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
cv.circle(img,(cx,cy), 20, (0,0, 255), 3)


# # Display the final image
cv.imshow('Gates Detected', img)
cv.waitKey(0)
cv.destroyAllWindows()


