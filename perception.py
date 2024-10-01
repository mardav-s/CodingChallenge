import cv2 as cv
import numpy as np
import scipy.optimize as optimize
from matplotlib import pyplot as plt

# Load image file
file_path = r"C:\Users\marda\Downloads\red.png"
image = cv.imread(file_path)

# Convert image to RGB and HSV formats
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Display the original RGB image
plt.subplot(1, 1, 1)
plt.imshow(image_rgb)
plt.show()

# Set up HSV thresholds for red hues
lower_red_range = cv.inRange(image_hsv, np.array([0, 135, 135]), np.array([15, 255, 255]))
upper_red_range = cv.inRange(image_hsv, np.array([159, 135, 135]), np.array([179, 255, 255]))

# Merge the two red hue masks
combined_red_mask = cv.bitwise_or(lower_red_range, upper_red_range)

# Apply morphological operations (erode and dilate) to remove noise
kernel_size = np.ones((5, 5))
cleaned_mask = cv.morphologyEx(combined_red_mask, cv.MORPH_OPEN, kernel_size)

# Blur the mask to further smooth the image
smoothed_mask = cv.medianBlur(cleaned_mask, 5)

# Detect edges in the processed image
edges = cv.Canny(smoothed_mask, 70, 255)
contours, _ = cv.findContours(np.array(edges), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_image = np.zeros_like(edges)

# Draw the contours on a blank image
cv.drawContours(contours_image, contours, -1, (255, 255, 255), 2)

# Simplify the contours using the Douglas-Peucker algorithm
simplified_contours = []
for contour in contours:
    simplified = cv.approxPolyDP(contour, 10, True)
    simplified_contours.append(simplified)
simplified_image = np.zeros_like(edges)
cv.drawContours(simplified_image, simplified_contours, -1, (255, 255, 255), 1)

# Compute the convex hulls for the simplified contours
convex_hulls = []
for contour in simplified_contours:
    convex_hulls.append(cv.convexHull(contour))
hulls_image = np.zeros_like(edges)
cv.drawContours(hulls_image, convex_hulls, -1, (255, 255, 255), 2)

# Keep only the convex hulls with 3 to 10 points
filtered_hulls = []
for hull in convex_hulls:
    if 3 <= len(hull) <= 10:
        filtered_hulls.append(hull)
filtered_hulls_image = np.zeros_like(edges)
cv.drawContours(filtered_hulls_image, filtered_hulls, -1, (255, 255, 255), 2)

# Function to check if a convex hull points upwards
# Divides the hull into top and bottom points relative to the vertical center of the hull
# Checks if the top points are within the horizontal bounds of the bottom points
def is_hull_pointing_up(hull):
    above_center = []
    below_center = []
    x, y, w, h = cv.boundingRect(hull)
    vertical_midline = y + h / 2
    aspect_ratio = w / h
    if aspect_ratio < 0.8:
        for point in hull:
            if point[0][1] < vertical_midline:
                above_center.append(point)
            else:
                below_center.append(point)
        left_bound = min(p[0][0] for p in below_center)
        right_bound = max(p[0][0] for p in below_center)
        for point in above_center:
            if not (left_bound <= point[0][0] <= right_bound):
                return False
    else:
        return False
    return True

# Identify cone-like shapes by analyzing the hulls
cones = []
bounding_boxes = []
for hull in filtered_hulls:
    if is_hull_pointing_up(hull):
        cones.append(hull)
        rect = cv.boundingRect(hull)
        bounding_boxes.append(rect)

# Draw the detected cones and their bounding boxes
cones_image = np.zeros_like(edges)
cv.drawContours(cones_image, cones, -1, (255, 255, 255), 2)
result_image = image_rgb.copy()
cv.drawContours(result_image, cones, -1, (255, 255, 255), 2)
for rect in bounding_boxes:
    cv.rectangle(result_image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (1, 255, 1), 3)

# Fit lines to the left and right sets of cone centers using least squares
def fit_line_least_squares(x, y):
    def line_equation(x, slope, intercept):
        return slope * x + intercept
    params, _ = optimize.curve_fit(line_equation, x, y)
    return params

# Determine points for line fitting on both sides of the image
left_cone_centers = [(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2) for rect in bounding_boxes if rect[0] + rect[2] / 2 < result_image.shape[1] / 2]
right_cone_centers = [(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2) for rect in bounding_boxes if rect[0] + rect[2] / 2 > result_image.shape[1] / 2]

# Compute the best-fit lines for both sides
slope_left, intercept_left = fit_line_least_squares(np.array([p[0] for p in left_cone_centers]), np.array([p[1] for p in left_cone_centers]))
slope_right, intercept_right = fit_line_least_squares(np.array([p[0] for p in right_cone_centers]), np.array([p[1] for p in right_cone_centers]))

# Draw the best-fit lines on the output image and save the result
cv.line(result_image, (0, int(intercept_left)), (3000, int(3000 * slope_left + intercept_left)), (255, 1, 1), 5)
cv.line(result_image, (0, int(intercept_right)), (3000, int(3000 * slope_right + intercept_right)), (255, 1, 1), 5)
plt.imshow(result_image)
plt.savefig("answer.png")
plt.show()