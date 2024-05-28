Tensor Spline API Call
######################

Example of using the TensorSpline API.

This example demonstrates how to create a basic interpolation using the TensorSpline API to interpolate multidimensional data.

Overview
========

This example script performs the following tasks:

1. Prepares random data samples and corresponding coordinates.
2. Sets up the TensorSpline with the specified bases and modes.
3. Creates evaluation coordinates extended and oversampled.
4. Performs standard evaluation on a grid of coordinates.
5. Evaluates using a meshgrid and compares the results with the standard evaluation.
6. Evaluates the tensor spline at a list of points and verifies the results.
7. Visualizes the original data samples and the interpolated data.

The main functions include:

- `main`:
  Demonstrates the usage of the TensorSpline API, including data preparation, tensor spline setup, evaluation, and visualization.

The code
========

.. literalinclude:: ../../examples/tensorspline_api_call.py
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
   from tensorspline_api_call import main
   main()
