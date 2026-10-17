"""Input/output helpers for TDSEZ."""

from zkit.io.base import detect_dimension
from zkit.io.eigen import read_eigen
from zkit.io.evolution import read_evolution
from zkit.io.kh import read_kh, read_multipole
from zkit.io.scid import read_scid_detail, read_scid_field, read_scid_pes
from zkit.io.surff import read_surff
from zkit.io.tdm import (
    read_tdm,
    tdm_magnitude_of_state,
    tdm_of_state,
)
from zkit.io.ts import read_timeseries
from zkit.io.wavefunction import read_wfs
from zkit.io.wfs_field import (
    compute_wfs_norm,
    reconstruct_static_wfs,
    reconstruct_wfs,
    wfs_to_vtk,
)

__all__ = [
    "read_kh",
    "read_multipole",
    "read_evolution",
    "read_eigen",
    "read_timeseries",
    "read_wfs",
    "detect_dimension",
    "read_tdm",
    "tdm_of_state",
    "tdm_magnitude_of_state",
    "reconstruct_wfs",
    "reconstruct_static_wfs",
    "wfs_to_vtk",
    "compute_wfs_norm",
    "read_surff",
    "read_scid_detail",
    "read_scid_field",
    "read_scid_pes",
]
