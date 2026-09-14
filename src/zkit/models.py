"""Data models for TDSEZ simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SimulationMeta:
    """Metadata parsed from the TDSEZ input file."""

    input_file: str
    dimension: int = 1
    n_splines: int = 0
    time_step: float = 0.0
    final_time: float = 0.0
    ns_energy: int = 0
    ns_population: int = 0
    output_stride_wfs: int = 0
    output_stride_ts: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        if self.time_step > 0:
            return int(round(self.final_time / self.time_step))
        return 0


@dataclass(frozen=True)
class TimeEvolution:
    """Time-dependent observables written by the TDSEZ monitor."""

    time: np.ndarray  # (n_steps,)
    dipoles: np.ndarray  # (n_steps, width)
    populations: np.ndarray  # (n_steps, 1 + n_populations)
    energies: np.ndarray  # (n_steps, 7)
    currents: np.ndarray  # (n_steps, 11)
    autocorrelation: np.ndarray  # (n_steps, 3)
    dimension: int = 1

    @property
    def n_steps(self) -> int:
        return int(self.time.shape[0])

    def __post_init__(self) -> None:
        if not isinstance(self.time, np.ndarray):
            raise TypeError("time must be a numpy.ndarray")
        if self.time.ndim not in (1, 2):
            raise ValueError(f"time must be 1-D or 2-D, got shape {self.time.shape}")

        arrays = [
            self.dipoles,
            self.populations,
            self.energies,
            self.currents,
            self.autocorrelation,
        ]
        names = ["dipoles", "populations", "energies", "currents", "autocorrelation"]
        for arr, name in zip(arrays, names):
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name!r} must be a numpy.ndarray, got {type(arr).__name__}")
            if arr.ndim != 2:
                raise ValueError(f"{name!r} must be 2-D, got shape {arr.shape}")
        rows = [arr.shape[0] for arr in arrays]
        if len(set(rows)) != 1:
            raise ValueError(
                f"All output arrays must share the same first dimension, got shapes: "
                f"{[arr.shape for arr in arrays]}"
            )


@dataclass(frozen=True)
class EigenData:
    """Eigenvalues and eigenvectors from the eigensolver."""

    values: np.ndarray  # (n_states,)
    vectors: np.ndarray  # (n_states, n_dof)
    n_states: int = 0
    dimension: int = 1
    _path: "str | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a numpy.ndarray")
        if not isinstance(self.vectors, np.ndarray):
            raise TypeError("vectors must be a numpy.ndarray")
        if self.values.ndim != 1:
            raise ValueError(f"values must be 1-D, got shape {self.values.shape}")
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {self.vectors.shape}")
        if self.values.shape[0] != self.vectors.shape[0]:
            raise ValueError("values and vectors must have the same leading dimension")

    def reconstruct(self, istate: int = 0, npoints: int = 200, axes=None):
        """Reconstruct eigenstate ``istate`` as a smooth real-space wavefunction.

        Delegates to :func:`zkit.io.reconstruct_static_wfs`, which evaluates the
        B-spline basis stored in the EigenData HDF5 file via igakit (the same
        engine PetIGA uses).  Requires the source file path (stored when read
        with :func:`zkit.read_eigen`).

        Returns a dict with keys ``axes, psi, Re, Im, abs2, SplineDegree,
        nfuncs, knots`` (see :func:`zkit.io.reconstruct_static_wfs`).
        """
        if self._path is None:
            raise RuntimeError(
                "EigenData._path is not set; reconstruct() needs the source "
                "EigenData_*.h5 path. Read it with zkit.read_eigen(path) first."
            )
        from zkit.io.wfs_field import reconstruct_static_wfs

        return reconstruct_static_wfs(self._path, istate=istate, npoints=npoints, axes=axes)


@dataclass(frozen=True)
class WavefunctionSeries:
    """Wavefunction snapshots saved by PETSc HDF5 viewer."""

    times: np.ndarray  # (n_snapshots,)
    data: np.ndarray  # (n_snapshots, n_dof)
    n_snapshots: int = 0
    dimension: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.times, np.ndarray):
            raise TypeError("times must be a numpy.ndarray")
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if self.times.ndim != 1:
            raise ValueError(f"times must be 1-D, got shape {self.times.shape}")
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2-D, got shape {self.data.shape}")
        if self.times.shape[0] != self.data.shape[0]:
            raise ValueError("times and data must have the same length")

    def __len__(self) -> int:
        return self.n_snapshots


@dataclass(frozen=True)
class TSSeries:
    """PETSc TS snapshot data (metadata only by default)."""

    times: np.ndarray  # (n_snapshots,)
    n_snapshots: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.times, np.ndarray):
            raise TypeError("times must be a numpy.ndarray")
        if self.times.ndim != 1:
            raise ValueError(f"times must be 1-D, got shape {self.times.shape}")


__all__ = [
    "SimulationMeta",
    "TimeEvolution",
    "EigenData",
    "WavefunctionSeries",
    "TSSeries",
]
