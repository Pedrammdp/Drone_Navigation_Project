import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read and Resize the image
img = cv.imread('Pics/test1-1.JPG')
resized = cv.resize(img, (750, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

kernel = np.ones((9,9),np.uint8)
# erosion = cv.erode(resized,kernel,iterations = 1)
# cv.imshow("erosion ",erosion )
dilation = cv.dilate(resized,kernel,iterations = 1)
cv.imshow("dilation",dilation )

# gradient = cv.morphologyEx(dilation, cv.MORPH_GRADIENT, kernel)
# cv.imshow("gradient",gradient)
# Convert to grayscale
gray = cv.cvtColor(dilation, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)

# Guassian Blur
gaussBl= cv.GaussianBlur(gray, (5,5),0)
cv.imshow('Gaussian blur', gaussBl)


# Calculate histograms
histr = cv.calcHist([gray],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()
intensity_values = np.array([x for x in range(histr.shape[0])])



color = ('blue','green','red')
for i,col in enumerate(color):
    histr = cv.calcHist([dilation],[i],None,[256],[0,256])
    plt.plot(intensity_values,histr,color = col,label=col+" channel")
    plt.xlim([0,256])
plt.legend()
plt.title("Histogram Channels")
plt.show()
# threshold the corner response
threshold, thresh = cv. threshold(gaussBl, 81, 255, cv.THRESH_BINARY)
cv.imshow('Simple Thresholded', thresh)

# Apply Canny edge detection
canny_edges = cv.Canny(thresh, 100, 300)
cv.imshow("canny", canny_edges)

# Find contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# Approximate contours
approx_contours = []
for cnt in contours:
    approx = cv.approxPolyDP(cnt, 0.02*cv.arcLength(cnt, True), True)
    if len(approx) == 4:
        approx_contours.append(approx)

# Find the rectangle
rect = cv.minAreaRect(approx_contours[0])

# Draw the rectangle
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(resized, approx_contours, -1, (0, 255, 0), 4)


# # Drawing circle at the center
cnt = approx_contours[0]
M = cv.moments(cnt)
print(M)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
cv.circle(resized,(cx,cy), 20, (0,0, 255), 3)

cv.imshow('Final result', resized)



cv.waitKey(0)