#!/usr/bin/env python3
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from scipy import ndimage, datasets

from bssp.interpolate.tensorspline import TensorSpline


def calculate_inscribed_rectangle_bounds_from_image(image):
    """
    Calculate the bounds for the largest rectangle that can be inscribed
    within a circle, which itself is inscribed within the original image,
    based on the image array directly.

    The rectangle and the circle are centered within the original image.

    Parameters:
    - image: The input image as a 2D or 3D numpy array.

    Returns:
    - A tuple (x_min, y_min, x_max, y_max) representing the bounds for cropping.
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
    Crop an image to the specified bounds.

    Parameters:
    - image: The input image as a 2D numpy array.
    - bounds: An array of (x_min, y_min, x_max, y_max) defining the crop bounds,
              where these values are absolute pixel coordinates in the image.

    Returns:
    - Cropped image as a 2D numpy array.
    """
    x_min, y_min, x_max, y_max = bounds
    return image[y_min:y_max, x_min:x_max]


def calculate_snr(original, modified):
    """
    Compute the Signal-to-Noise Ratio (SNR) between the original and modified images.

    Parameters:
    - original: The original image as a 2D numpy array.
    - modified: The modified (rotated) image as a 2D numpy array.

    Returns:
    - SNR value as a float.
    """
    original_normalized = original / 255.0 if original.max() > 1 else original
    processed_normalized = modified / 255.0 if modified.max() > 1 else modified
    noise = original_normalized - processed_normalized
    mean_signal = np.mean(original_normalized)
    variance_noise = np.var(noise)
    epsilon = 1e-3
    snr = 10 * np.log10((mean_signal**2) / (variance_noise + epsilon))
    return snr


def calculate_mse(original, modified):
    """
    Compute the Mean Squared Error (MSE) between the original and modified images.

    Parameters:
    - original: The original image as a 2D numpy array.
    - modified: The modified (rotated) image as a 2D numpy array.

    Returns:
    - MSE value as a float.
    """
    mse = np.mean((original - modified) ** 2)
    return mse


def rotate_image_and_crop_bssp(image, angle, order=3, mode="zero", iterations=1):
    """
    Rotate an image by a specified angle using the BSSP library's TensorSpline method and crop the result.

    Parameters:
    - image: The input image as a 2D numpy array.
    - angle: The rotation angle in degrees.
    - order: The order of the spline (0-7).
    - mode: The mode for handling boundaries (default is "zero").
    - iterations: The number of iterations to apply the rotation.

    Returns:
    - Rotated image as a 2D numpy array.
    """
    dtype = image.dtype
    ny, nx = image.shape
    xx = np.linspace(0, nx - 1, nx, dtype=dtype)
    yy = np.linspace(0, ny - 1, ny, dtype=dtype)
    data = np.ascontiguousarray(image, dtype=dtype)

    rotated_image_cp = cp.asarray(data)
    xx_cp = cp.asarray(xx)
    yy_cp = cp.asarray(yy)

    order = max(0, min(order, 7))
    basis = f"bspline{order}"

    for _ in range(iterations):
        tensor_spline = TensorSpline(
            data=rotated_image_cp, coordinates=(yy_cp, xx_cp), bases=basis, modes=mode
        )
        angle_rad = np.radians(-angle)
        cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
        original_center_x, original_center_y = (nx - 1) / 2.0, (ny - 1) / 2.0
        oy, ox = cp.ogrid[0:ny, 0:nx]
        ox = ox - original_center_x
        oy = oy - original_center_y

        nx_coords = cos_angle * ox + sin_angle * oy + original_center_x
        ny_coords = -sin_angle * ox + cos_angle * oy + original_center_y

        eval_coords_cp = cp.asarray(ny_coords.flatten()), cp.asarray(
            nx_coords.flatten()
        )
        interpolated_values_cp = tensor_spline(coordinates=eval_coords_cp, grid=False)
        rotated_image_cp = interpolated_values_cp.reshape(ny, nx)

    rotated_image = rotated_image_cp.get()
    cp.get_default_memory_pool().free_all_blocks()
    return rotated_image


def rotate_image_and_crop_scipy(image, angle, order=3, iterations=5):
    """
    Rotate an image by a specified angle using SciPy's ndimage.rotate function and crop the result.

    Parameters:
    - image: The input image as a 2D numpy array.
    - angle: The rotation angle in degrees.
    - order: The order of the spline (0-5).
    - iterations: The number of iterations to apply the rotation.

    Returns:
    - Rotated image as a 2D numpy array.
    """
    rotated_image = image.copy()
    for _ in range(iterations):
        rotated_image = ndimage.rotate(
            rotated_image, angle, reshape=False, order=order, mode="constant", cval=0
        )
    return rotated_image


def benchmark_and_display_rotation(image, angle, order, iterations):
    """
    Perform a benchmark of the rotation operation for both BSSP and SciPy libraries and display images.

    Parameters:
    - image: The input image as a 2D numpy array.
    - angle: The rotation angle in degrees.
    - order: The order of the spline (0-7).
    - iterations: The number of iterations to apply the rotation.
    """
    start_time_custom = time.time()
    custom_rotated_and_cropped_bssp = rotate_image_and_crop_bssp(
        image, angle, order=order, mode="zero", iterations=iterations
    )
    time_custom = time.time() - start_time_custom

    start_time_scipy = time.time()
    scipy_rotated_and_cropped = rotate_image_and_crop_scipy(
        image, angle, order=order, iterations=iterations
    )
    time_scipy = time.time() - start_time_scipy

    bounds = calculate_inscribed_rectangle_bounds_from_image(image)
    image_cropped = crop_image_to_bounds(image, bounds)
    custom_rotated_and_cropped_bssp = crop_image_to_bounds(
        custom_rotated_and_cropped_bssp, bounds
    )
    scipy_rotated_and_cropped = crop_image_to_bounds(scipy_rotated_and_cropped, bounds)

    snr_bssp = calculate_snr(image_cropped, custom_rotated_and_cropped_bssp)
    snr_scipy = calculate_snr(image_cropped, scipy_rotated_and_cropped)
    mse_bssp = calculate_mse(image_cropped, custom_rotated_and_cropped_bssp)
    mse_scipy = calculate_mse(image_cropped, scipy_rotated_and_cropped)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axes[0].imshow(image_cropped, cmap="gray")
    axes[0].set_title("Original Image")
    axes[1].imshow(custom_rotated_and_cropped_bssp, cmap="gray")
    axes[1].set_title(
        f"BSSP Rotated\nSNR: {snr_bssp:.2f}dB, MSE: {mse_bssp:.2e}\nAngle: {angle}°, Iter: {iterations}\nOrder: {order}, Time: {time_custom:.2f}s"
    )
    axes[2].imshow(scipy_rotated_and_cropped, cmap="gray")
    axes[2].set_title(
        f"SciPy Rotated\nSNR: {snr_scipy:.2f}dB, MSE: {mse_scipy:.2e}\nAngle: {angle}°, Iter: {iterations}\nOrder: {order}, Time: {time_scipy:.2f}s"
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.15)
    plt.show()


def main():
    """
    Main function to load the image, perform rotations using both BSSP and SciPy methods,
    and display the results.
    """
    # Image size, Rotation angle and iterations and order of spline interpolation
    size = 1000
    angle = 72  # 72
    iterations = 5  # 5
    order = 3

    # Load and resize the ascent image
    image = datasets.ascent()
    image_resized = ndimage.zoom(
        image, (size / image.shape[0], size / image.shape[1]), order=order
    )

    # Convert to float32
    image_resized = image_resized.astype(np.float32)

    # Benchmark and display rotation results
    benchmark_and_display_rotation(image_resized, angle, order, iterations)


if __name__ == "__main__":
    main()
