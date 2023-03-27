import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('Pics/test2-2.jpg')
resized = cv.resize(img, (750, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Convert to HSV color space
hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)
cv.imshow('hsv', hsv)
cv.imwrite('hsv.png', hsv)

# Threshold the image based on the gate color
lower_color = np.array([0, 155, 0])  # replace with the lower HSV values of the gate color
upper_color = np.array([179, 215, 110])  # replace with the upper HSV values of the gate color
mask = cv.inRange(hsv, lower_color, upper_color)

# Apply Gaussian blur to the image 
blurred = cv.GaussianBlur(mask, (7, 7), 0)
cv.imshow('blurred', blurred)

# Apply morphological operations to fill gaps in the gate contour
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
closed = cv.morphologyEx(blurred, cv.MORPH_CLOSE, kernel)
open = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)
eroded = cv.erode(open, kernel, iterations=1)
dilated = cv.dilate(eroded, kernel, iterations=1)
cv.imshow('Open', open)
cv.imshow('Eroded', eroded)
cv.imshow('Dilated', dilated)

# Apply Canny edge detection to detect the edges of the gate
edges = cv.Canny(blurred, 100, 300)
cv.imshow('edges', edges)

# Find contours in the edge image
contours, heirarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

for i, contour in enumerate(contours):
    area = cv.contourArea(contour)
    print(f"Contour {i+1} area: {area}")
area_threshold_min = 100

# # Loop through the contours and erase the small ones
contours_filtered = []
for contour in contours:
    area = cv.contourArea(contour)
    if area > area_threshold_min:
        contours_filtered.append(contour)



# # Draw each contour in a separate window
for i, contour in enumerate(contours_filtered):
    # Create a new window for this contour
    window_name = f'Contour {i+1}'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 500, 500)

#     # Draw the contour on the window
    contour_img = cv.drawContours(np.zeros_like(resized), [contour], 0, (0, 255, 0), 2)
    cv.imshow(window_name, contour_img)


# Draw circle at center of each contour in a new window
for i, contour in enumerate(contours_filtered):
    # Calculate center of contour using moments
    M = cv.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
#     # Create a new window for the contour
    cv.namedWindow(f'contour_{i+1}', cv.WINDOW_NORMAL)
    cv.resizeWindow(f'contour_{i+1}', 300, 300)
    
#     # Draw circle at center of contour in new window
    new_img = resized.copy()
    cv.circle(new_img, (cx, cy), 20, (0, 0, 255), 3)
    cv.imshow(f'contour_{i+1}', new_img)






cv.waitKey(0)
cv.destroyAllWindows()