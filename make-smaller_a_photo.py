import cv2
import numpy as np

# Load your image
img = cv2.imread('path_to_your_image.jpg')

# Convert to grayscale for easier processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a threshold to isolate bees
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours which will correspond to bees
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def crop_and_resize(image, contour, size=128):
    # Compute the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    # Crop the image to the bounding box
    cropped_image = image[y:y+h, x:x+w]
    # Resize to 128x128
    resized_image = cv2.resize(cropped_image, (size, size))
    return resized_image

# Process each contour
cropped_images = []
for contour in contours:
    if cv2.contourArea(contour) > 100:  # Filter out too small contours
        cropped_image = crop_and_resize(img, contour)
        cropped_images.append(cropped_image)

# Optionally, save or display your images
for i, cropped_image in enumerate(cropped_images):
    cv2.imwrite(f'bee_{i}.jpg', cropped_image)
