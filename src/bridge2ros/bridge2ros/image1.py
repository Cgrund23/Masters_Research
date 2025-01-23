import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the PNG image
image_path = 'track_image.png'
image = cv2.imread(image_path)

# Step 2: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply thresholding to binarize the image (adjust threshold values as needed)
_, thresh_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)

# Step 4: Find all contours (both inner and outer)
contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image to draw the contours
contour_image = np.zeros_like(image)

# Step 5: Draw the contours on a blank canvas (for visualization)
cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 5)
output_path = 'track_borders.png'
cv2.imwrite(output_path, contour_image)
# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(contour_image)
plt.title('Inner and Outer Borders')
plt.axis('off')
plt.show()

# If you want the points of the polyline borders for further processing:
polyline_points = [contour.reshape(-1, 2) for contour in contours]  # Extract points (x, y)