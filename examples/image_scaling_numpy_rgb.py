import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from bssp.interpolate.tensorspline import TensorSpline
import time

import math


def load_image(file_path):
    """
    Load an image from the specified file path.
    """
    return plt.imread(file_path).astype(np.float32)


def scale_image_bssp(image, scale_row, scale_col, order=3, mode="mirror"):
    dtype = image.dtype
    ny, nx, channels = image.shape
    scaled_images = []

    for ch in range(channels):
        data = np.ascontiguousarray(image[:, :, ch], dtype=dtype)
        xx = np.linspace(0, nx - 1, nx, dtype=dtype)
        yy = np.linspace(0, ny - 1, ny, dtype=dtype)

        # Tensor spline setup for each channel
        bases = "bspline" + str(order)
        tensor_spline = TensorSpline(
            data=data, coordinates=(yy, xx), bases=bases, modes=mode
        )

        # Create evaluation coordinates (scaling the coordinates)
        eval_xx = np.linspace(0, nx - 1, int(nx * scale_col), dtype=dtype)
        eval_yy = np.linspace(0, ny - 1, int(ny * scale_row), dtype=dtype)
        eval_coords = (eval_yy, eval_xx)

        # Evaluate the tensor spline on the new grid for each channel
        data_eval = tensor_spline(coordinates=eval_coords)
        scaled_images.append(
            data_eval.reshape((int(ny * scale_row), int(nx * scale_col)))
        )

    # Stack the processed channels back into a single RGB image
    scaled_image_rgb = np.stack(scaled_images, axis=-1)

    # Ensure the image data is in the range 0 to 255
    scaled_image_rgb = np.clip(scaled_image_rgb, 0, 255)

    return scaled_image_rgb


# def save_image(file_path, image):
#    """
#    Save the specified image to the specified file path, ensuring it's in the correct format.
#    Explicitly scales `uint8` images from [0, 1] to [0, 255].
#    """
#    # Check if the image data is `uint8` and not using the full range
#    if image.dtype == np.uint8 and np.max(image) <= 1:
#        # The image data is in the range 0 to 1, scale it up to 0 to 255
#        image = (image * 255).astype(np.uint8)
#
#    plt.imsave(file_path, image, format="png")


def display_image(image):
    """
    Display the specified image using matplotlib and enable manual saving through the GUI.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray" if len(image.shape) == 2 else None)
    plt.axis("off")  # Turn off axis numbering
    # plt.title("Scaled Image")
    plt.show()


# Parameters
scale_row = 3.0
scale_col = 3.0
image_file_path = "C:\\Users\\Pablo\\Desktop\\EPFL\\bssp\\docs\\_static\\logo.png"
output_file_path = (
    "C:\\Users\\Pablo\\Desktop\\EPFL\\bssp\\docs\\_static\\scaled_logo.png"
)

# Load image
image = load_image(image_file_path)

# Measure and scale with bssp
start_bssp = time.time()
scaled_image_bssp = scale_image_bssp(image, scale_row, scale_col)
end_bssp = time.time()
bssp_time = end_bssp - start_bssp

# Save the scaled image
# save_image(output_file_path, scaled_image_bssp)
display_image(scaled_image_bssp)

# Display the original and scaled images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axs[0].imshow(image, cmap="gray", interpolation="nearest")
axs[0].set_title("Original Image")
axs[0].axis("on")

# BSSP scaled image
axs[1].imshow(scaled_image_bssp, cmap="gray", interpolation="nearest")
axs[1].set_title(f"BSSP Scaled Image\nTime: {bssp_time:.4f} s")
axs[1].axis("on")

plt.show()
