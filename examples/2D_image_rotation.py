#!/usr/bin/env python3

"""
This script performs two main analyses on image rotation using both our custom library (BSSP)
and SciPy's ndimage.map_coordinates for comparison.

1) For a specific spline order, the script rotates an image by 360 degrees through 36 iterations 
of 10 degrees each. It compares the performance (time spent and Signal-to-Noise Ratio (SNR)) 
between the BSSP library and SciPy.

2) The script repeats the first analysis with different spline orders ranging from 0 to 5. It then plots 
the time spent versus SNR for each order, allowing for a visual comparison between the BSSP 
library and SciPy's implementation across different spline orders.
"""

import numpy as np
from scipy import ndimage, datasets
import matplotlib.pyplot as plt
import time

from bssp.interpolate.tensorspline import TensorSpline

def calculate_inscribed_rectangle_bounds_from_image(image):
    """
    Calculate the bounds for the largest rectangle that can be inscribed within a circle, 
    which itself is inscribed within the original image, based on the image array directly.
    
    Parameters:
    - image (numpy.ndarray): The input image as a 2D or 3D numpy array. The image is assumed to be square.
    
    Returns:
    - numpy.ndarray: A 1D array of bounds (x_min, y_min, x_max, y_max) for the inscribed rectangle.
    """
    # Extract image dimensions
    height, width = image.shape[:2]
    
    # Calculate the radius of the inscribed circle
    radius = min(width, height) / 2
    
    # The side length of the square (largest inscribed rectangle in a circle)
    side_length = radius * np.sqrt(2)
    
    # Calculate the center of the image
    cx, cy = width / 2, height / 2
    
    # Calculate the bounds of the largest inscribed rectangle
    x_min = int(cx - side_length / 2)
    y_min = int(cy - side_length / 2)
    x_max = int(cx + side_length / 2)
    y_max = int(cy + side_length / 2)
    
    return np.array([x_min, y_min, x_max, y_max])

def crop_image_to_bounds(image, bounds):
    """
    Crop an image to the specified bounds, defined by minimum and maximum coordinates.
    
    Parameters:
    - image (numpy.ndarray): The input image as a 2D numpy array.
    - bounds (numpy.ndarray): An array of (x_min, y_min, x_max, y_max) defining the crop bounds,
                              where these values are absolute pixel coordinates in the image.
    
    Returns:
    - numpy.ndarray: The cropped image as a 2D numpy array.
    """
    x_min, y_min, x_max, y_max = bounds

    # Directly use the provided bounds to crop the image
    return image[y_min:y_max, x_min:x_max]

def calculate_snr(original, modified):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for comparing the original and modified images.
    
    Parameters:
    - original (numpy.ndarray): The original image before modification.
    - modified (numpy.ndarray): The image after modification.
    
    Returns:
    - float: The SNR value in decibels (dB).
    """
    signal_power = np.mean(original**2)
    noise_power = np.mean((original - modified)**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def rotate_image_and_crop_bssp(image, angle, order=3, mode='zero', iterations=5):
    """
    Rotate an image by a specified angle using the BSSP library's TensorSpline method and crop the result.
    
    Parameters:
    - image (numpy.ndarray): The input image to rotate.
    - angle (float): Rotation angle in degrees.
    - order (int): The order of the spline interpolation. Default is 3.
    - mode (str): The mode of extrapolation. Default is 'zero'.
    - iterations (int): The number of times the rotation is applied. Default is 5.
    
    Returns:
    - numpy.ndarray: The rotated and cropped image.
    """
    dtype = "float32"
    ny, nx = image.shape
    xx = np.linspace(0, nx-1, nx, dtype=dtype)
    yy = np.linspace(0, ny-1, ny, dtype=dtype)
    data = np.ascontiguousarray(image, dtype=dtype)
    rotated_image = data.copy()
    
    # Ensure the order falls within 0 to 7
    order = max(0, min(order, 7))

    # Format the basis string based on the order
    basis = f"bspline{order}"

    for _ in range(iterations):
        tensor_spline = TensorSpline(data=rotated_image, coordinates=(yy, xx), bases=basis, modes=mode)
        angle_rad = np.radians(-angle)
        cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
        
        original_center_x, original_center_y = (nx - 1) / 2.0, (ny - 1) / 2.0
        oy, ox = np.ogrid[0:ny, 0:nx]
        ox = ox - original_center_x
        oy = oy - original_center_y

        nx_coords = cos_angle * ox + sin_angle * oy + original_center_x
        ny_coords = -sin_angle * ox + cos_angle * oy + original_center_y

        eval_coords = (ny_coords.flatten(), nx_coords.flatten())
        interpolated_values = tensor_spline(coordinates=eval_coords, grid=False)

        rotated_image = interpolated_values.reshape(ny, nx)
    
    return rotated_image

def rotate_image_and_crop_scipy(image, angle, order=3, iterations=5):
    """
    Rotate an image by a specified angle using SciPy's ndimage.rotate function and crop the result.
    
    Parameters:
    - image (numpy.ndarray): The input image to rotate.
    - angle (float): Rotation angle in degrees.
    - order (int): The order of the spline interpolation. Default is 3.
    - iterations (int): The number of times the rotation is applied. Default is 5.
    
    Returns:
    - numpy.ndarray: The rotated and cropped image.
    """
    rotated_image = image.copy()
    for _ in range(iterations):
        rotated_image = ndimage.rotate(rotated_image, angle, reshape=False, order=order, mode='constant', cval=0)
    return rotated_image

# Load and resize the ascent image to 256 x 256
image = datasets.ascent()
image_resized = ndimage.zoom(image, (256/image.shape[0], 256/image.shape[1]), order=3)

# Convert to float32
image_resized = image_resized.astype(np.float32)

# Rotation angle and iterations
angle = 10 #10
iterations = 36 #36

# Comparison for a particular order
order = 5

# Custom Rotation using BSSP
start_time_custom = time.time()
custom_rotated_and_cropped_bssp = rotate_image_and_crop_bssp(image_resized, angle, order=order, mode='zero', iterations=iterations)
time_custom = time.time() - start_time_custom

# SciPy Rotation
start_time_scipy = time.time()
scipy_rotated_and_cropped = rotate_image_and_crop_scipy(image_resized, angle, order=order, iterations=iterations)
time_scipy = time.time() - start_time_scipy

# Crop original and rotated images to properly compute SNR for both methods
bounds = calculate_inscribed_rectangle_bounds_from_image(image_resized)
image_resized_and_cropped = crop_image_to_bounds(image_resized, bounds)
custom_rotated_and_cropped_bssp = crop_image_to_bounds(custom_rotated_and_cropped_bssp, bounds)
scipy_rotated_and_cropped = crop_image_to_bounds(scipy_rotated_and_cropped, bounds)

# Calculate SNR for both methods
snr_bssp = calculate_snr(image_resized_and_cropped, custom_rotated_and_cropped_bssp)
snr_scipy = calculate_snr(image_resized_and_cropped, scipy_rotated_and_cropped)

# Display the original, custom rotated (using BSSP), and SciPy rotated (and cropped) images with SNR
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
axes[0].imshow(image_resized_and_cropped, cmap='gray')
axes[0].set_title("Original Image after Cropping)")
axes[1].imshow(custom_rotated_and_cropped_bssp, cmap='gray')
axes[1].set_title(f"BSSP Rotated and Cropped\nSNR: {snr_bssp:.2f}dB\nOrder: {order}\nIterations: {iterations}\nAngle: {angle}°\nTime: {time_custom:.2f}s")
axes[2].imshow(scipy_rotated_and_cropped, cmap='gray')
axes[2].set_title(f"SciPy Rotated and Cropped\nSNR: {snr_scipy:.2f}dB\nOrder: {order}\nIterations: {iterations}\nAngle: {angle}°\nTime: {time_scipy:.2f}s")

plt.tight_layout()
# Modified to prevent blocking
plt.show(block=False)

# Analysis of SNR and time performance for all orders

# Initialize lists to store results
orders = list(range(6))  # Spline orders from 0 to 5
times_bssp = []
snrs_bssp = []
times_scipy = []
snrs_scipy = []

for order in orders:
    # BSSP Rotation
    start_time = time.time()
    custom_rotated = rotate_image_and_crop_bssp(image_resized, angle, order=order, iterations=iterations)
    time_bssp = time.time() - start_time
    custom_rotated_and_cropped = crop_image_to_bounds(custom_rotated, bounds)
    snr_bssp = calculate_snr(image_resized_and_cropped, custom_rotated_and_cropped)
    
    # SciPy Rotation
    start_time = time.time()
    scipy_rotated = rotate_image_and_crop_scipy(image_resized, angle, order=order, iterations=iterations)
    time_scipy = time.time() - start_time
    scipy_rotated_and_cropped = crop_image_to_bounds(scipy_rotated, bounds)
    snr_scipy = calculate_snr(image_resized_and_cropped, scipy_rotated_and_cropped)

    # Store results
    times_bssp.append(time_bssp)
    snrs_bssp.append(snr_bssp)
    times_scipy.append(time_scipy)
    snrs_scipy.append(snr_scipy)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(times_bssp, snrs_bssp, '-o', label='BSSP', color='blue')
plt.plot(times_scipy, snrs_scipy, '-o', label='SciPy', color='red')

# Label data points with increased font size
for i, order in enumerate(orders):
    plt.text(times_bssp[i], snrs_bssp[i], str(order), color='blue', fontsize=12)  # Adjust fontsize as needed
    plt.text(times_scipy[i], snrs_scipy[i], str(order), color='red', fontsize=12)  # Adjust fontsize as needed

plt.xlabel('Time (seconds)')
plt.ylabel('SNR (dB)')
plt.title('Comparison of BSSP vs SciPy Rotations (36 times 10 degree rotation) by Spline Order')
plt.legend()
plt.grid(True)

# Final plt.show() call to ensure all windows stay open
plt.show()