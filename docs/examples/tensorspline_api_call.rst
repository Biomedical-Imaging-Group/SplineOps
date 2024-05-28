Tensor Spline API Call
######################

Below is an example of using `TensorSpline` to interpolate multidimensional data. 
This example demonstrates setting up the tensor spline, creating evaluation coordinates, 
and generating plots for both the original and interpolated data.

.. literalinclude:: ../../examples/tensorspline_api_call.py
   :language: python
   :linenos:

Executing the Example
=====================

The following code executes the example script and generates the plots:

.. plot::
   :context: reset
   :include-source: false

   import sys
   import os
   sys.path.insert(0, os.path.abspath('../../examples'))
   from tensorspline_api_call import main
   main()
