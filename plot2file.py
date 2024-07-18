# parameters:
# x and y sampling
# dealing with time, dates, labels

import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('filename')

#args = parser.parse_args()

#img = cv2.imread(args.filename)
img = cv2.imread("./data/example.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width = img.shape[:2]
# this value determines which lines are eligible to be considered as an axis
# a value of e.g. 0.3 means that a line's length must be at least 30% of the image size to be a candidate for an axis
axis_threshold = 0.3
line_threshold_x = width * axis_threshold
line_threshold_y = height * axis_threshold
line_threshold = min(line_threshold_x, line_threshold_y)
# fudge factor to determine wether a line is axis aligned
# a value of e.g. 5 means that a line may differ only 5° from the imgage x or y axes
axis_aligned_threshold = 2

kernel_size = 5
img_gray_blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)

low_t = 50
high_t = 150
edges = cv2.Canny(img_gray_blur, low_t, high_t)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = line_threshold  # minimum number of pixels making up a line
max_line_gap = 10  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        angle = abs(angle % 180)
        if abs(angle) <= axis_aligned_threshold or abs(angle - 90) <= axis_aligned_threshold:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3)
        else:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),3)

img_lines = cv2.addWeighted(img, 0.8, line_image, 1, 0)

show = img_lines
cv2.imshow('temp', line_image)
cv2.waitKey(0)