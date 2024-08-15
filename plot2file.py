# TODO:
# split up axes
# x and y sampling
# dealing with time, dates, labels
# logarithmic scale
# Outlier removal

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('filename')

#args = parser.parse_args()

# 6
#x_val_min = -1360
#x_val_max = -980
#y_val_min = -1550
#y_val_max = -1040
# 2
x_val_min = -1500
x_val_max = 2000
y_val_min = 0
y_val_max = 1


#img = cv2.imread(args.filename)
img = cv2.imread("./data/dataset_2.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width = img.shape[:2]
# this value determines which lines are eligible to be considered as an axis
# a value of e.g. 0.3 means that a line's length must be at least 30% of the image size to be a candidate for an axis
axis_threshold = 0.4
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

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

x_lines = []
y_lines = []

for line in lines:
    for x1,y1,x2,y2 in line:
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        angle = abs(angle % 180)
        if abs(angle) <= axis_aligned_threshold:
            x_lines.append((x1, y1, x2, y2))
        elif abs(angle - 90) <= axis_aligned_threshold:
            y_lines.append((x1, y1, x2, y2))
        else:
            pass
            #cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 16)

# find the most extreme coordinates for the axes
x_axis = max(x_lines, key=lambda ele: ele[1])
y_axis = min(y_lines, key=lambda ele: ele[0])
x_axis_y = x_axis[1]
y_axis_x = y_axis[0]
# sometimes, the lines in x/y_axis don't stretch along the entire graph
# therefore, we need to find other lines at similar coordinates and amalgate the result into a synthetic line
x_axis_y_stretch = x_axis_y * 0.05
y_axis_x_stretch = y_axis_x * 0.05
x_axis_min = width
x_axis_max = 0
y_axis_min = height
y_axis_max = 0
for line in x_lines:
    x1, y1, x2, y2 = line
    if y1 > x_axis_y - x_axis_y_stretch and y1 < x_axis_y + x_axis_y_stretch:
        x_axis_min = min(x_axis_min, x1)
        x_axis_max = max(x_axis_max, x2)
for line in y_lines:
    x1, y1, x2, y2 = line
    if x1 > y_axis_x - y_axis_x_stretch and x1 < y_axis_x + y_axis_x_stretch:
        y_axis_min = min(y_axis_min, y1)
        y_axis_max = max(y_axis_max, y2)
x_axis_min, x_axis_max = min(x_axis_min, x_axis_max), max(x_axis_min, x_axis_max)
y_axis_min, y_axis_max = min(y_axis_min, y_axis_max), max(y_axis_min, y_axis_max)
cv2.line(line_image, [x_axis_min, x_axis_y], [x_axis_max, x_axis_y], (32, 240, 160), 1)
cv2.line(line_image, [y_axis_x, y_axis_min], [y_axis_x, y_axis_max], (32, 240, 160), 1)

lines = cv2.HoughLinesP(edges, rho, theta, 5, np.array([]), max_line_gap, max_line_gap)

line_equations = []
for line in lines:
    for x1, y1, x2, y2 in line:
        x1_t, x2_t = min(x1, x2), max(x1, x2)
        y1_t, y2_t = min(y1, y2), max(y1, y2)
        if x1_t > x_axis_min and x2_t < x_axis_max and y1_t > y_axis_min and y2_t < y_axis_max:
            cv2.line(line_image, [x1, y1], [x2, y2], (255, 0, 0), 1)
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            line_equations.append((x1_t, x2_t, a, b))

# for every single pixel along the x axis, we check all the lines that we know of, calculate y values and take the max (== closest pixel to the x axis)
# this way we can get an accurate estimate even if the line detection algorithm produced overlapping lines
x_values = list(range(x_axis_min, x_axis_max + 1))
y_values = []
for x in x_values:
    temp_max = 0
    for mi, ma, a, b in line_equations:
        if x >= mi and x <= ma:
            temp_max = max(temp_max, a * x + b)
    y_values.append(-temp_max if temp_max else None)

1==1
x_values = [((x_val_max - x_val_min) * (x - x_axis_min) / (x_axis_max - x_axis_min)) + x_val_min for x in x_values]
1 == 1
y_values = [((y_val_max - y_val_min) * y / (y_axis_max - y_axis_min)) + y_val_min + (y_val_max - y_val_min) if y else None for y in y_values]

cv2.imshow("tt", line_image)

plt.plot(x_values, y_values)
plt.show()