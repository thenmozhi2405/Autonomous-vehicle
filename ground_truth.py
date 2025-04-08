import cv2
import numpy as np

# Load the image
image_path = "C:/Users/Admin/Desktop/test.jpg"  # Path to the uploaded image
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    raise FileNotFoundError("Error: Image not found. Check file path.")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to smooth the image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Otsu's Thresholding to create a binary mask
_, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save the ground truth mask
output_path = "C:/Users/Admin/Desktop/ground_truth_mask.png"
cv2.imwrite(output_path, binary_mask)

# Display the mask
cv2.imshow("Ground Truth Mask", binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Ground truth mask saved at: {output_path}")
