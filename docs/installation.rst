Installation
============

Requirements
------------

- Python ``>= 3.10``
- ``numpy >= 1.23``
- ``h5py >= 3.8``

Optional features:

- ``sympy`` for deck-based physics extraction with ``zkit.physics``
- B-spline reconstruction is included in the core package through the bundled
  ``igakit.igalib.bsp`` compatibility evaluator
- ``pyvista`` for VTK ParaView exports
- ``torch >= 1.12`` for the ``zkit.mwigner`` Wigner-transform subpackage

Install from source
-------------------

.. code-block:: bash

   cd /path/to/zkit
   pip install -e .

The package name on PyPI is ``zkit-lib``, but the import name stays ``zkit``.

Install optional features
-------------------------

.. code-block:: bash

   pip install -e ".[physics,viz,vtk,mwigner]"

The ``reconstruction`` extra is retained as an empty compatibility extra for
older installation commands; no external igakit package is required.

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
