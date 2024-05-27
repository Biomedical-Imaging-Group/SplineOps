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
    image = np.random.uniform(low=0, high=255, size=(size, size)).astype(np.float32)
    blurred_image = ndimage.gaussian_filter(image, sigma=3)
    blurred_image = blurred_image.astype(np.float32)
    return blurred_image


def calculate_inscribed_rectangle_bounds_from_image(image):
    height, width = image.shape[:2]
    radius = min(width, height) / 2
    side_length = radius * np.sqrt(2)
    cx, cy = width / 2, height / 2
    x_min = int(cx - side_length / 2)
    y_min = int(cy - side_length / 2)
    x_max = int(cx + side_length / 2)
    y_max = int(cy + side_length / 2)
    return np.array([x_min, y_min, x_max, y_max])


def crop_image_to_bounds(image, bounds):
    x_min, y_min, x_max, y_max = bounds
    return image[y_min:y_max, x_min:x_max]


def calculate_snr(original, processed):
    original_normalized = original / 255.0 if original.max() > 1 else original
    processed_normalized = processed / 255.0 if processed.max() > 1 else processed
    noise = original_normalized - processed_normalized
    mean_signal = np.mean(original_normalized)
    variance_noise = np.var(noise)
    epsilon = 1e-3
    snr = 10 * np.log10((mean_signal**2) / (variance_noise + epsilon))
    return snr


def rotate_image_and_crop_bssp(image, angle, order=3, mode="zero", iterations=1):
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
        cp.get_default_memory_pool().free_all_blocks()

    return rotated_image


def rotate_image_and_crop_scipy(image, angle, order=3, iterations=1):
    rotated_image = image.copy()
    for _ in range(iterations):
        rotated_image = ndimage.rotate(
            rotated_image, angle, reshape=False, order=order, mode="constant", cval=0
        )
    return rotated_image


def benchmark_and_display_rotation(size, angle, order, iterations):
    image = generate_artificial_image(size)
    rotated_bssp = rotate_image_and_crop_bssp(
        image, angle, order=order, iterations=iterations
    )
    rotated_scipy = rotate_image_and_crop_scipy(
        image, angle, order=order, iterations=iterations
    )
    snr_bssp = calculate_snr(image, rotated_bssp)
    snr_scipy = calculate_snr(image, rotated_scipy)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Original Image")
    axs[1].imshow(rotated_bssp, cmap="gray")
    axs[1].set_title(f"BSSP Rotated Image\nSNR: {snr_bssp:.2f} dB")
    axs[2].imshow(rotated_scipy, cmap="gray")
    axs[2].set_title(f"SciPy Rotated Image\nSNR: {snr_scipy:.2f} dB")
    for ax in axs:
        ax.axis("off")
    plt.show()


def benchmark_rotation(size, angle, order, iterations):
    image = generate_artificial_image(size)
    rotate_image_and_crop_bssp(image, angle=angle, order=order, iterations=1)
    result_bssp = cupyx.profiler.benchmark(
        rotate_image_and_crop_bssp,
        (image, angle, order, "zero", iterations),
        n_repeat=1,
    )
    start_cpu_scipy = time.perf_counter()
    rotate_image_and_crop_scipy(image, angle=angle, order=order, iterations=iterations)
    end_cpu_scipy = time.perf_counter()
    time_scipy = end_cpu_scipy - start_cpu_scipy
    return size, result_bssp.cpu_times.mean(), time_scipy


def main():
    image_sizes = [10, 50, 100]
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
    benchmark_and_display_rotation(
        size=100, angle=angle, order=order, iterations=iterations
    )


if __name__ == "__main__":
    main()
