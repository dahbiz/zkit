Wavefunctions
=============

zkit reads two kinds of wavefunction outputs:

1. Time-series snapshots from ``td/wfs_<input>.h5``
2. Static eigenstates from ``static/EigenData_<input>.h5``

Both contain IGA B-spline coefficients, not real-space arrays. zkit evaluates
the basis with igakit so you can plot or export on a regular grid.

Read snapshots
--------------

.. code-block:: python

   from zkit.io.wavefunction import read_wfs

   wfs = read_wfs("td/wfs_h2p.inp.h5")
   print("snapshots:", wfs.n_snapshots)
   print("data shape:", wfs.data.shape)
   print("dimension:", wfs.dimension)

``wfs.data`` is ``(n_snapshots, n_dof)`` complex coefficients in the IGA tensor
order used by PetIGA.

Reconstruct a snapshot
----------------------

.. code-block:: python

   from zkit.io.wfs_field import reconstruct_wfs

   res = reconstruct_wfs(
       "td/wfs_h2p.inp.h5",
       step=0,
       npoints=240,
       axes=None,
   )

   print(res["dim"])
   print(res["psi"].shape)
   print(res["axes"][0].min(), res["axes"][0].max())

Returned keys:

- ``dim``
- ``step``
- ``axes``
- ``psi``
- ``Re``
- ``Im``
- ``abs2``
- ``SplineDegree``
- ``nfuncs``
- ``knots``

Reconstruct a static eigenstate
-------------------------------

.. code-block:: python

   from zkit.io.eigen import read_eigen

   eigen = read_eigen("static/EigenData_h2p.inp.h5")
   res = eigen.reconstruct(istate=0, npoints=240)
   print(res["psi"].shape)

Exact norm via Gauss quadrature
-------------------------------

.. code-block:: python

   from zkit.io.wfs_field import compute_wfs_norm

   norm = compute_wfs_norm("td/wfs_h2p.inp.h5", step=0, nq=5)
   print(norm)

``nq=5`` is exact for spline degree <= 9. Higher degrees need larger ``nq``.

VTK export
----------

.. code-block:: python

   from zkit.io.wfs_field import reconstruct_wfs, wfs_to_vtk

   res = reconstruct_wfs("td/wfs_h2p.inp.h5", step=0, npoints=120)
   wfs_to_vtk(res, "figures/wfs.vts")

Supported extensions:

- ``.vts`` StructuredGrid
- ``.vtu`` UnstructuredGrid
- ``.vtk`` legacy StructuredGrid
