Installation
============

Requirements
------------

- Python ``>= 3.10``
- ``numpy >= 1.23``
- ``h5py >= 3.8``

Optional extras:

- ``sympy`` for deck-based physics extraction with ``zkit.physics``
- ``igakit`` for B-spline reconstruction from IGA coefficients
- ``pyvista`` for VTK ParaView exports
- ``torch >= 1.12`` for the ``zkit.mwigner`` Wigner-transform subpackage

Install from source
-------------------

.. code-block:: bash

   cd /path/to/zkit
   pip install -e .

The package name on PyPI is ``zkit-lib``, but the import name stays ``zkit``.

Editable install with extras
----------------------------

.. code-block:: bash

   pip install -e ".[physics,reconstruction,viz,vtk,mwigner]"

Without installing
------------------

.. code-block:: bash

   export PYTHONPATH=/path/to/zkit/src:$PYTHONPATH
   python -m zkit --help

Verify the installation
-----------------------

.. code-block:: python

   import zkit
   print("version:", zkit.__version__)
