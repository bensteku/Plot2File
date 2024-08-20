# plot2file

This is a small tool to extract the original data from x-y line plots with a single graph. Several solutions for this already exist, both paid and free as well as AI- and traditional CV-based. I wanted to make one for myself though. This one uses OpenCV's Hough Line Transform.

## Usage

To run the script pandas, OpenCV, numpy and matplotlib are needed. It can be called like this:

```
python plot2file.py (image of plot) (name of output csv) xmin xmax ymin ymax [--interactive] [--show_plot] [--rho] [--theta] [--max_line_gap] [--min_line_length] [--line_votes]
```

- 'xmin', 'xmax', 'ymin' and 'ymax' are the values on the respective axes on the image of the plot
- '--interactive' is an optional command. If set, the user will be able to mark the location of the x-axis, the y-axis, the upper edge of the plot and the right edge of the plot (in that order) themselves via a double-click. If not set, the program will use some heuristics to approximate the position of these lines by itself. Naturally, placing the lines manually will tend to create more accurate results. It also enables you to extract just portions of the data, given that the xmin etc. parameters are adapted correctly.
- '--show_plot' will display a plot based on the generated x-y data. This is useful to get a rough impression of the accuracy of the conversion.
- '--rho' and all the parameters behind it are optionals related to the parameters of the [OpenCV Hough Lines Transform](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html).

The output will be dumped into a csv file with an x and y column. 