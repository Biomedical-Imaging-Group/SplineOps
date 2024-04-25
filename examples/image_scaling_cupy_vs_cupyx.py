import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from bssp.interpolate.tensorspline import TensorSpline
import time
import math
import cupyx.scipy.ndimage


def generate_artificial_image(size):
    image = np.random.uniform(low=0, high=255, size=(size, size)).astype(np.float32)
    blurred_image = ndimage.gaussian_filter(image, sigma=3)
    return blurred_image.astype(np.float32)


def warm_up_and_time_gpu(
    image, scale_row, scale_col, order=3, mode="mirror", iterations=5
):
    dtype = image.dtype
    ny, nx = image.shape
    data_cp = cp.asarray(image)
    xx_cp = cp.linspace(0, nx - 1, nx, dtype=dtype)
    yy_cp = cp.linspace(0, ny - 1, ny, dtype=dtype)
    tensor_spline_cp = TensorSpline(
        data=data_cp, coordinates=(yy_cp, xx_cp), bases=f"bspline{order}", modes=mode
    )

    eval_xx_cp = cp.linspace(0, nx - 1, int(nx * scale_col), dtype=dtype)
    eval_yy_cp = cp.linspace(0, ny - 1, int(ny * scale_row), dtype=dtype)
    eval_coords_cp = (eval_yy_cp, eval_xx_cp)

    # Warm-up run
    _ = tensor_spline_cp(coordinates=eval_coords_cp)
    del _

    # Timed runs
    times = []
    for _ in range(iterations):
        start_time = time.time()
        data_eval_cp = tensor_spline_cp(coordinates=eval_coords_cp)
        scaled_image_cp = data_eval_cp.reshape(
            (int(ny * scale_row), int(nx * scale_col))
        )
        cp.cuda.Stream.null.synchronize()  # Ensure GPU computation is complete
        end_time = time.time()
        times.append(end_time - start_time)
        del data_eval_cp  # Free memory immediately

    scaled_image_np = cp.asnumpy(scaled_image_cp)
    del scaled_image_cp  # Free memory immediately
    cp.get_default_memory_pool().free_all_blocks()
    return scaled_image_np, np.mean(times)


def scale_image_ndimage(image, scale_row, scale_col, iterations=5):
    dtype = image.dtype
    image_cp = cp.asarray(image)  # Convert the image to CuPy array for GPU processing
    times = []
    for _ in range(iterations):
        start_time = time.time()
        scaled_image_cp = cupyx.scipy.ndimage.zoom(
            image_cp, (scale_row, scale_col), order=3
        )
        cp.cuda.Stream.null.synchronize()  # Ensure GPU computation is complete
        end_time = time.time()
        times.append(end_time - start_time)
        if _ == iterations - 1:  # Convert to NumPy only after the last iteration
            scaled_image_np = cp.asnumpy(scaled_image_cp)
        del scaled_image_cp  # Free GPU memory immediately after use

    cp.get_default_memory_pool().free_all_blocks()  # Free any unused memory blocks
    return scaled_image_np, np.mean(times)


# Rest of the script remains unchanged


size = 1000
scale_row = math.pi
scale_col = math.pi
image_np = generate_artificial_image(size)
scaled_image_gpu, time_gpu = warm_up_and_time_gpu(image_np, scale_row, scale_col)
scaled_image_nd, time_nd = scale_image_ndimage(image_np, scale_row, scale_col)

print(f"Average BSSP Time: {time_gpu:.4f} seconds")
print(f"Average cupyx.scipy.ndimage Time: {time_nd:.4f} seconds")

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(image_np, cmap="gray", interpolation="nearest")
axs[0].set_title("Original Image")
axs[0].axis("on")
axs[0].grid(True)

axs[1].imshow(scaled_image_gpu, cmap="gray", interpolation="nearest")
axs[1].set_title(f"BSSP Scaled Image\nTime: {time_gpu:.4f} s")
axs[1].axis("on")
axs[1].grid(True)

axs[2].imshow(scaled_image_nd, cmap="gray", interpolation="nearest")
axs[2].set_title(f"Cupyx.scipy.ndimage Scaled Image\nTime: {time_nd:.4f} s")
axs[2].axis("on")
axs[2].grid(True)

plt.show()
