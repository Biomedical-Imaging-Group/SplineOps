Installation
============

Install minimal dependencies in a dedicated environment (shown here using Mamba).

First, activate your environment:

.. code-block:: bash

    mamba activate <env-name>

Minimal requirements:

.. code-block:: bash

    mamba install numpy scipy

Simply install bssp from its wheel using pip. **IMPORTANT**: Not yet uploaded on pypi or anaconda/mamba. A wheel is needed and can be obtained from the source (see Packaging below).

.. code-block:: bash

    pip install bssp

To run the examples, matplotlib will also be required.

.. code-block:: bash

    mamba install matplotlib
