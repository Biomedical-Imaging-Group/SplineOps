GPU Interoperability Using CuPy
###############################

Example of using the TensorSpline API with GPU support.

This example demonstrates how to create a basic interpolation using the TensorSpline API from the BSSP library, leveraging both CPU (NumPy) and GPU (CuPy) computations.

Overview
========

This example script performs the following tasks:

1. Generates random data samples and corresponding coordinates.
2. Creates tensor splines using both NumPy (CPU) and CuPy (GPU).
3. Evaluates the tensor splines on a set of extended and oversampled coordinates.
4. Computes the absolute difference between the evaluations obtained from NumPy and CuPy.
5. Plots the results, including the evaluations and their absolute difference.

The main steps include:

- Generating random data samples and coordinates using NumPy.
- Creating tensor splines from these data samples for both CPU (NumPy) and GPU (CuPy) computations.
- Evaluating the tensor splines on a set of coordinates.
- Calculating the maximum absolute difference between the evaluations from NumPy and CuPy.
- Visualizing the results using Matplotlib.

The code
========

.. literalinclude:: ../../examples/gpu_interoperability_using_cupy.py
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
   from gpu_interoperability_using_cupy import main
   main()
