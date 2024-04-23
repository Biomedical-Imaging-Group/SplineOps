import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from bssp.interpolate.tensorspline import TensorSpline
import time

import math


def generate_artificial_image(size):
    """
    Generate an artificial image with uniform noise and blur it.
    """
    image = np.random.uniform(low=0, high=255, size=(size, size)).astype(np.float32)
    blurred_image = ndimage.gaussian_filter(image, sigma=3)  # Apply Gaussian blur
    return blurred_image.astype(np.float32)


def scale_image_bssp(image, scale_row, scale_col, order=3, mode="mirror"):
    dtype = image.dtype
    ny, nx = image.shape
    xx = np.linspace(0, nx - 1, nx, dtype=dtype)
    yy = np.linspace(0, ny - 1, ny, dtype=dtype)
    data = np.ascontiguousarray(image, dtype=dtype)

    # Tensor spline setup
    bases = "bspline" + str(order)
    tensor_spline = TensorSpline(
        data=data, coordinates=(yy, xx), bases=bases, modes=mode
    )

    # Create evaluation coordinates (scaling the coordinates)
    eval_xx = np.linspace(0, nx - 1, int(nx * scale_col), dtype=dtype)
    eval_yy = np.linspace(0, ny - 1, int(ny * scale_row), dtype=dtype)
    eval_coords = (eval_yy, eval_xx)

    # Evaluate the tensor spline on the new grid
    data_eval = tensor_spline(coordinates=eval_coords)

    return data_eval.reshape((int(ny * scale_row), int(nx * scale_col)))


def scale_image_ndimage(image, scale_row, scale_col, order=3):
    """
    Scale an image using scipy.ndimage.zoom function.
    """
    return ndimage.zoom(
        image, (scale_row, scale_col), order=order
    )  # Using cubic spline interpolation


# Parameters
size = 1000
scale_row = math.pi
scale_col = math.pi
image = generate_artificial_image(size)

# Measure and scale with bssp
start_bssp = time.time()
scaled_image_bssp = scale_image_bssp(image, scale_row, scale_col)
end_bssp = time.time()
bssp_time = end_bssp - start_bssp

# Measure and scale with ndimage
start_ndimage = time.time()
scaled_image_ndimage = scale_image_ndimage(image, scale_row, scale_col)
end_ndimage = time.time()
ndimage_time = end_ndimage - start_ndimage

scaled_size_x = int(size * scale_col)
scaled_size_y = int(size * scale_row)

# Display the original, bssp scaled, and ndimage scaled images
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Original image
axs[0].imshow(image, cmap="gray", interpolation="nearest")
axs[0].set_title("Original Image")
axs[0].axis("on")
axs[0].set_xticks(np.linspace(0, size, num=6))
axs[0].set_yticks(np.linspace(0, size, num=6))
axs[0].grid(True)

# BSSP scaled image
axs[1].imshow(scaled_image_bssp, cmap="gray", interpolation="nearest")
axs[1].set_title(f"BSSP Scaled Image\nTime: {bssp_time:.4f} s")
axs[1].axis("on")
axs[1].set_xticks(np.linspace(0, scaled_size_x, num=6))
axs[1].set_yticks(np.linspace(0, scaled_size_y, num=6))
axs[1].grid(True)

# Ndimage scaled image
axs[2].imshow(scaled_image_ndimage, cmap="gray", interpolation="nearest")
axs[2].set_title(f"ndimage.zoom Scaled Image\nTime: {ndimage_time:.4f} s")
axs[2].axis("on")
axs[2].set_xticks(np.linspace(0, scaled_size_x, num=6))
axs[2].set_yticks(np.linspace(0, scaled_size_y, num=6))
axs[2].grid(True)

plt.show()
