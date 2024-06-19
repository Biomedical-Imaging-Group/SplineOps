Formatting, Type Checking, and Testing
======================================

Formatting and type checking is performed using the following commands:

.. code-block:: bash

    tox -e format
    tox -e type

Testing with the following command:

.. code-block:: bash

    tox

**IMPORTANT**: Since CI is not implemented, make sure to run, pass, and/or fix:

.. code-block:: bash

    tox -e format
    tox -e type
    tox
