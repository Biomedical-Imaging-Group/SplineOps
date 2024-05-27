"""
Example of using the TensorSpline API.

This example demonstrates how to create a basic interpolation using the
TensorSpline API.
"""

#!/usr/bin/env python3
import numpy as np
import cupy as cp
import cupyx.profiler
from scipy import ndimage
import matplotlib.pyplot as plt
import time

from bssp.interpolate.tensorspline import TensorSpline


def generate_artificial_image(size):
    """
    Generate an artificial image with uniform noise and blur it.
    """
    image = np.random.uniform(low=0, high=255, size=(size, size)).astype(np.float32)
    blurred_image = ndimage.gaussian_filter(image, sigma=3)  # Apply Gaussian blur
    blurred_image = blurred_image.astype(
        np.float32
    )  # Ensure the blurred image is float32
    return blurred_image


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


def calculate_snr(original, processed):
    """
    Compute the Signal-to-Noise Ratio (SNR) between the original and processed images,
    ensuring both are on the same scale and data type.
    """
    # Ensure both images are in the same range, e.g., 0 to 1
    original_normalized = original / 255.0 if original.max() > 1 else original
    processed_normalized = processed / 255.0 if processed.max() > 1 else processed

    # Compute noise as the difference
    noise = original_normalized - processed_normalized

    # Calculate mean of the original signal and variance of the noise
    mean_signal = np.mean(original_normalized)
    variance_noise = np.var(noise)

    # A small epsilon to prevent division by zero
    epsilon = 1e-3

    # Compute SNR, adding epsilon to the denominator
    snr = 10 * np.log10((mean_signal**2) / (variance_noise + epsilon))
    return snr


def rotate_image_and_crop_bssp(image, angle, order=3, mode="zero", iterations=1):
    """
    Rotate an image by a specified angle using the BSSP library's TensorSpline method and crop the result.
    """
    dtype = image.dtype
    ny, nx = image.shape
    xx = np.linspace(0, nx - 1, nx, dtype=dtype)
    yy = np.linspace(0, ny - 1, ny, dtype=dtype)
    data = np.ascontiguousarray(image, dtype=dtype)

    rotated_image_cp = cp.asarray(data)
    xx_cp = cp.asarray(xx)
    yy_cp = cp.asarray(yy)

    # Ensure the order falls within 0 to 7
    order = max(0, min(order, 7))

    # Format the basis string based on the order
    basis = f"bspline{order}"

    for _ in range(iterations):
        tensor_spline = TensorSpline(
            data=rotated_image_cp, coordinates=(yy_cp, xx_cp), bases=basis, modes=mode
        )
        angle_rad = np.radians(-angle)
        cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)

        original_center_x, original_center_y = (nx - 1) / 2.0, (ny - 1) / 2.0
        oy, ox = np.ogrid[0:ny, 0:nx]
        ox = ox - original_center_x
        oy = oy - original_center_y

        nx_coords = cos_angle * ox + sin_angle * oy + original_center_x
        ny_coords = -sin_angle * ox + cos_angle * oy + original_center_y

        eval_coords_cp = cp.asarray(ny_coords.flatten()), cp.asarray(
            nx_coords.flatten()
        )
        interpolated_values_cp = tensor_spline(coordinates=eval_coords_cp, grid=False)
        interpolated_values_np = interpolated_values_cp.get()

        rotated_image = interpolated_values_np.reshape(ny, nx)

        # Clean up memory explicitly if necessary (not always required if using memory pool)
        cp.get_default_memory_pool().free_all_blocks()

    return rotated_image


def rotate_image_and_crop_scipy(image, angle, order=3, iterations=1):
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
        rotated_image = ndimage.rotate(
            rotated_image, angle, reshape=False, order=order, mode="constant", cval=0
        )
    return rotated_image


def benchmark_and_display_rotation(size, angle, order, iterations):
    """
    Perform a benchmark of the rotation operation for both BSSP and SciPy libraries and display images.
    """
    image = generate_artificial_image(size)

    # Perform rotations
    rotated_bssp = rotate_image_and_crop_bssp(
        image, angle, order=order, iterations=iterations
    )  # Adjust with actual BSSP rotation
    rotated_scipy = rotate_image_and_crop_scipy(
        image, angle, order=order, iterations=iterations
    )

    # Compute SNR
    snr_bssp = calculate_snr(image, rotated_bssp)
    snr_scipy = calculate_snr(image, rotated_scipy)

    # Display images and SNR
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Original Image")
    axs[1].imshow(rotated_bssp, cmap="gray")  # Adjust this according to BSSP's output
    axs[1].set_title(f"BSSP Rotated Image\nSNR: {snr_bssp:.2f} dB")
    axs[2].imshow(rotated_scipy, cmap="gray")
    axs[2].set_title(f"SciPy Rotated Image\nSNR: {snr_scipy:.2f} dB")
    for ax in axs:
        ax.axis("off")
    plt.show()


"""
def benchmark_rotation(size, angle, order, iterations):
    ""
    Perform a benchmark of the rotation operation for both BSSP and SciPy libraries.
    ""
    image = generate_artificial_image(size)

    # BSSP Rotation (GPU accelerated)
    start_gpu_bssp = cp.cuda.Event()
    end_gpu_bssp = cp.cuda.Event()
    start_gpu_bssp.record()  # Start GPU timer for BSSP
    start_cpu_bssp = time.perf_counter()  # Start CPU timer for BSSP
    # Assuming rotate_image_and_crop_bssp works with cupy arrays and is GPU-accelerated
    rotate_image_and_crop_bssp(image, angle=angle, order=order, iterations=iterations)
    end_cpu_bssp = time.perf_counter()  # End CPU timer for BSSP
    end_gpu_bssp.record()  # End GPU timer for BSSP
    end_gpu_bssp.synchronize()  # Wait for GPU operation to complete
    time_bssp = end_cpu_bssp - start_cpu_bssp  # Calculate elapsed CPU time for BSSP

    # SciPy Rotation (CPU only)
    start_cpu_scipy = time.perf_counter()  # Start CPU timer for SciPy
    rotate_image_and_crop_scipy(image, angle=angle, order=order, iterations=iterations)
    end_cpu_scipy = time.perf_counter()  # End CPU timer for SciPy
    time_scipy = end_cpu_scipy - start_cpu_scipy  # Calculate elapsed CPU time for SciPy

    return size, time_bssp, time_scipy
"""


def benchmark_rotation(size, angle, order, iterations):
    """
    Perform a benchmark of the rotation operation for both BSSP and SciPy libraries.
    """
    image = generate_artificial_image(size)

    # Warm-up (important for JIT)
    rotate_image_and_crop_bssp(image, angle=angle, order=order, iterations=1)

    # Benchmark using cupyx.profiler
    result_bssp = cupyx.profiler.benchmark(
        rotate_image_and_crop_bssp,
        (image, angle, order, "zero", iterations),
        n_repeat=1,  # Number of repeats to average over
    )

    # SciPy Rotation (CPU only) - continue using time.perf_counter for CPU benchmarks
    start_cpu_scipy = time.perf_counter()
    rotate_image_and_crop_scipy(image, angle=angle, order=order, iterations=iterations)
    end_cpu_scipy = time.perf_counter()
    time_scipy = end_cpu_scipy - start_cpu_scipy

    return size, result_bssp.cpu_times.mean(), time_scipy


# List of image sizes to benchmark
image_sizes = [10, 50, 100]
# image_sizes = [100, 500, 1000, 5000, 10000]
# image_sizes = [1000, 2000, 3000, 4000, 5000]
# Placeholder lists to store benchmark results
sizes = []
times_bssp = []
times_scipy = []

angle = 360
order = 3
iterations = 1

for size in image_sizes:
    size, time_bssp, time_scipy = benchmark_rotation(
        size, angle=angle, order=order, iterations=iterations
    )
    sizes.append(size)
    times_bssp.append(time_bssp)
    times_scipy.append(time_scipy)

# Plotting the benchmark results
plt.figure(figsize=(10, 6))
plt.plot(sizes, times_bssp, "-o", label="BSSP", color="blue")
plt.plot(sizes, times_scipy, "-o", label="SciPy", color="red")
plt.xlabel("Image Size (pixels)")
plt.ylabel("Time (seconds)")
plt.title("Benchmark: BSSP vs SciPy Image Rotation Time by Image Size")
plt.legend()
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.show(block=False)

# Example usage for a single size
benchmark_and_display_rotation(
    size=100, angle=angle, order=order, iterations=iterations
)
