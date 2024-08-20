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
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('filename')
parser.add_argument('output')
parser.add_argument('x_min', type=float)
parser.add_argument('x_max', type=float)
parser.add_argument('y_min', type=float)
parser.add_argument('y_max', type=float)
parser.add_argument('--interactive', action='store_true')
parser.add_argument('--show_plot', action='store_true')
# Hough line detection parameters
parser.add_argument('--rho', type=int, default=1)
parser.add_argument('--theta', type=float, default=np.pi / 180)
parser.add_argument('--max_line_gap', type=int, default=10)
parser.add_argument('--min_line_length', type=int, default=20)
parser.add_argument('--line_votes', type=int, default=5)
args = parser.parse_args()

mode = 0  # mode for the interactive bbox determination
bbox = [0, 0, 0, 0]  # x min, x max, y min, y max

img = cv2.imread(args.filename)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel_size = 5
gray_blur_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
low_t = 50
high_t = 150
edge_img = cv2.Canny(gray_blur_img, low_t, high_t)
show_img = img.copy() * 0
tmp_img = img.copy() * 0

# helper method for the interactive mode
def draw_axes(event, x, y, flags, param):
    global bbox, mode, tmp_img, img
    tmp_img = img.copy()
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if mode == 0:  # placing x axis
            cv2.line(img, (0, y), (img.shape[1], y), (255, 0, 0), 3)
            bbox[3] = y
            mode = 1
        elif mode == 1:  # placing y axis
            cv2.line(img, (x, 0), (x, img.shape[0]), (0, 255, 0), 3)
            bbox[0] = x
            mode = 2
        elif mode == 2:  # placing upper x axis
            cv2.line(img, (0, y), (img.shape[1], y), (255, 0, 0), 3)
            bbox[2] = y
            mode = 3
        elif mode == 3:  # placing right y axis
            cv2.line(img, (x, 0), (x, img.shape[0]), (0, 255, 0), 3)
            bbox[1] = x
            mode = 4
    # live drawing
    if mode == 0 or mode == 2:
        cv2.line(tmp_img, (0, y), (img.shape[1], y), (255, 0, 0), 3)
    else:
        cv2.line(tmp_img, (x, 0), (x, img.shape[0]), (0, 255, 0), 3)

def automatic_axes_detection(axis_length_threshold=0.4,
                             axis_aligned_threshold = 2):
    global bbox
    height, width = img.shape[:2]
    # this determines which lines are eligible to be considered as an axis
    # an axis_threshold value of e.g. 0.3 means that a line's length must be at least 30% of the image size to be a candidate for an axis
    line_threshold_x = width * axis_length_threshold
    line_threshold_y = height * axis_length_threshold
    line_threshold = min(line_threshold_x, line_threshold_y)

    lines = cv2.HoughLinesP(edge_img, args.rho, args.theta, 15, np.array([]), line_threshold, args.max_line_gap)

    x_lines = []
    y_lines = []

    # find lines that are close enough to being aligned with either x- or y-axis
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

    # find the most extreme coordinates for the axes
    x_axis = max(x_lines, key=lambda ele: ele[1])
    y_axis = min(y_lines, key=lambda ele: ele[0])
    x_axis_y = x_axis[1]
    y_axis_x = y_axis[0]
    # sometimes, the lines in x/y_axis don't stretch along the entire graph
    # therefore, we need to find other lines at similar coordinates and amalgate the result into a synthetic line
    x_axis_y_stretch = x_axis_y * 0.05
    y_axis_x_stretch = y_axis_x * 0.05
    x_axis_min_px = width
    x_axis_max_px = 0
    y_axis_min_px = height
    y_axis_max_px = 0
    for line in x_lines:
        x1, y1, x2, y2 = line
        if y1 > x_axis_y - x_axis_y_stretch and y1 < x_axis_y + x_axis_y_stretch:
            x_axis_min_px = min(x_axis_min_px, x1)
            x_axis_max_px = max(x_axis_max_px, x2)
    for line in y_lines:
        x1, y1, x2, y2 = line
        if x1 > y_axis_x - y_axis_x_stretch and x1 < y_axis_x + y_axis_x_stretch:
            y_axis_min_px = min(y_axis_min_px, y1)
            y_axis_max_px = max(y_axis_max_px, y2)
    x_axis_min_px, x_axis_max_px = min(x_axis_min_px, x_axis_max_px), max(x_axis_min_px, x_axis_max_px)
    y_axis_min_px, y_axis_max_px = min(y_axis_min_px, y_axis_max_px), max(y_axis_min_px, y_axis_max_px)
    bbox = [x_axis_min_px, x_axis_max_px, y_axis_min_px, y_axis_max_px]

# bbox determination
if args.interactive:
    cv2.namedWindow('plot')
    cv2.setMouseCallback('plot', draw_axes)

    while(mode != 4):
        #cv2.addWeighted(img, 0.5, tmp_img, 0.5, 0, show_img)
        cv2.imshow('plot', tmp_img)
        k = cv2.waitKey(20) & 0xFF
    print("Bounding box of plot from user input (px):")
else:
    automatic_axes_detection()
    print("Bounding box from automatic determination (px):")
print(bbox)

# rerun the line detection but more fine-grained
lines = cv2.HoughLinesP(edge_img, args.rho, args.theta, args.line_votes, np.array([]), args.min_line_length, args.max_line_gap)

# throw out every line that isn't within the box defined by the x and y axes
# and find linear equations for the rest of the lines
line_equations = []
for line in lines:
    for x1, y1, x2, y2 in line:
        x1_t, x2_t = min(x1, x2), max(x1, x2)
        y1_t, y2_t = min(y1, y2), max(y1, y2)
        if x1_t > bbox[0] and x2_t < bbox[1] and y1_t > bbox[2] and y2_t < bbox[3]:
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            line_equations.append((x1_t, x2_t, a, b))

# for every single pixel along the x axis, we check all the line equations that we know of, calculate y values and take the max (== closest pixel to the x axis)
# this way we can get an accurate estimate even if the line detection algorithm produced overlapping lines
x_values = list(range(bbox[0], bbox[1] + 1))
y_values = []
for x in x_values:
    temp_max = 0
    for mi, ma, a, b in line_equations:
        if x >= mi and x <= ma:
            temp_max = max(temp_max, a * x + b)
    y_values.append(-temp_max if temp_max else None)

# convert pixel values into actual data values
x_values = [((args.x_max - args.x_min) * (x - bbox[0]) / (bbox[1] - bbox[0])) + args.x_min for x in x_values]
y_values = [((args.y_max - args.y_min) * y / (bbox[3] - bbox[2])) + args.y_min + (args.y_max - args.y_min) if y else None for y in y_values]

if args.show_plot:
    plt.plot(x_values, y_values)
    plt.show()

df = pd.DataFrame({'x': x_values, 'y': y_values})
df = df.dropna()
df.to_csv(args.output, index=False)