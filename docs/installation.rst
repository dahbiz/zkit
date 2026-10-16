Installation
============

Requirements
------------

- Python ``>= 3.10``
- ``numpy >= 1.23``
- ``h5py >= 3.8``

Optional features (extras):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Adds
   * - ``viz``
     - plotting (matplotlib)
   * - ``vtk``
     - VTK export of wavefunctions (pyvista)
   * - ``physics``
     - symbolic field expressions in ``zkit.physics`` (sympy)
   * - ``validation``
     - the ``zkit validate`` suite (scipy, sympy)
   * - ``mwigner``
     - the ``zkit.mwigner`` Wigner-function subpackage (PyTorch >= 2.0, matplotlib)
   * - ``docs``
     - building this documentation (Sphinx)

B-spline reconstruction is part of the core package (a bundled pure-Python
``igakit.igalib.bsp`` evaluator); no external igakit is required.

Install
-------

.. code-block:: bash

   pip install "git+https://github.com/dahbiz/zkit.git"

or, from a clone (editable, with optional features):

.. code-block:: bash

   git clone https://github.com/dahbiz/zkit.git
   cd zkit
   pip install -e ".[viz,validation]"

The distribution name is ``zkit-lib``; the import name and the command are
``zkit``.

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
