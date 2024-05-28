2D Image Rotation
##################

Example of using the TensorSpline API.

This example demonstrates how to create a basic interpolation using the TensorSpline API to rotate a 2D image and compare it with SciPy's rotation functionality.

Overview
========

This example script performs the following tasks:

1. Loads and resizes a sample image.
2. Rotates the image using both the TensorSpline API from the BSSP library and SciPy's ndimage.rotate function.
3. Crops the rotated images to the largest inscribed rectangle to remove boundary artifacts.
4. Calculates the Signal-to-Noise Ratio (SNR) and Mean Squared Error (MSE) between the original and rotated images.
5. Displays the original and rotated images with relevant metrics.

The main functions include:

- `calculate_inscribed_rectangle_bounds_from_image`:
  Calculates the bounds for cropping the image to the largest inscribed rectangle.

- `crop_image_to_bounds`:
  Crops the image based on the specified bounds.

- `calculate_snr`:
  Computes the Signal-to-Noise Ratio (SNR) between the original and rotated images.

- `calculate_mse`:
  Computes the Mean Squared Error (MSE) between the original and rotated images.

- `rotate_image_and_crop_bssp`:
  Rotates the image using the TensorSpline method from the BSSP library.

- `rotate_image_and_crop_scipy`:
  Rotates the image using SciPy's ndimage.rotate function.

- `benchmark_and_display_rotation`:
  Benchmarks both rotation methods, calculates metrics, and displays the results.

The code
========

.. literalinclude:: ../../examples/image_rotation_2D.py
   :language: python
   :linenos:

Executing the code
==================

The following code executes the example script and generates the plots:

.. plot::
   :context: reset
   :include-source: false

   import sys
   import os
   sys.path.insert(0, os.path.abspath('../../examples'))
   from image_rotation_2D import main
   main()
