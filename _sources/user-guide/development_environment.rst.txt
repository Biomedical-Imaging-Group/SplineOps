Development Environment
=======================

Easiest way to install dev dependencies:

.. code-block:: bash

    mamba install cupy numpy scipy black mypy tox hatch pytest matplotlib

Install bssp in editable mode:

.. code-block:: bash

    pip install -e .

If a specific CUDA version is required:

.. code-block:: bash

    mamba install cupy cuda-version=12.3

Potential other CuPy libraries (CuPy from Conda-Forge):

.. code-block:: bash

    mamba install cupy cutensor cudnn nccl
