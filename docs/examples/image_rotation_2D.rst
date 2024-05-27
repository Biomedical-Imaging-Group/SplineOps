2D Image Rotation
##################

Example of using the TensorSpline API.

This example demonstrates how to create a basic interpolation using the
TensorSpline API.

.. literalinclude:: ../../examples/image_rotation_2D.py
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
   from image_rotation_2D import main
   main()
