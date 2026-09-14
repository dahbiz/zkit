"""High-level simulation reader for TDSEZ."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

if TYPE_CHECKING:
    import numpy as np

    from zkit.physics import Physics

from zkit.io.base import detect_dimension
from zkit.io.eigen import read_eigen
from zkit.io.evolution import read_evolution
from zkit.io.ts import read_timeseries
from zkit.io.wavefunction import read_wfs
from zkit.models import EigenData, SimulationMeta, TimeEvolution, TSSeries, WavefunctionSeries


class Run:
    """Read all outputs of a TDSEZ simulation run.

    Parameters
    ----------
    run_dir:
        Directory containing the ``td/`` and ``static/`` subdirectories.
    basename:
        Input basename such as ``h2p.inp`` or full path to the input file.

    Examples
    --------
    >>> run = zkit.Run("build", "h2p.inp")
    >>> run.dim
    1
    >>> run.dipoles.shape
    (101, 4)
    """

    def __init__(
        self,
        run_dir: Union[str, Path],
        basename: Union[str, Path],
    ) -> None:
        self.run_dir = Path(run_dir)
        self.input_file = Path(basename)
        self.base = self.input_file.name
        self.td_dir = self.run_dir / "td"
        self.static_dir = self.run_dir / "static"

        self._meta: Optional[SimulationMeta] = None
        self._evolution: Optional[TimeEvolution] = None
        self._eigen: Optional[EigenData] = None
        self._wavefunctions: Optional[WavefunctionSeries] = None
        self._ts: Optional[TSSeries] = None
        self._physics = None

    @property
    def meta(self) -> SimulationMeta:
        if self._meta is None:
            self._meta = self._parse_meta()
        return self._meta

    @property
    def dim(self) -> int:
        # Prefer the data itself (works even when the input file is absent);
        # fall back to the parsed input-file metadata.
        try:
            ev_path = self.td_dir / f"TimeEvolutionData_{self.base}.h5"
            if ev_path.exists():
                return detect_dimension(ev_path)
            eig_path = self.static_dir / f"EigenData_{self.base}.h5"
            if eig_path.exists():
                return detect_dimension(eig_path)
            wfs_path = self.td_dir / f"wfs_{self.base}.h5"
            if wfs_path.exists():
                return detect_dimension(wfs_path)
        except (ValueError, OSError):
            pass
        return self.meta.dimension

    @property
    def evolution(self) -> TimeEvolution:
        if self._evolution is None:
            path = self.td_dir / f"TimeEvolutionData_{self.base}.h5"
            # Detect dimension from the data; only fall back to the
            # input-file parse if the file carries no usable signal.
            try:
                dim = detect_dimension(path)
            except (ValueError, OSError):
                dim = self.meta.dimension
            self._evolution = read_evolution(path, dimension=dim)
        return self._evolution

    @property
    def spectrum(self) -> EigenData:
        if self._eigen is None:
            path = self.static_dir / f"EigenData_{self.base}.h5"
            self._eigen = read_eigen(path)
        return self._eigen

    @property
    def wavefunctions(self) -> WavefunctionSeries:
        if self._wavefunctions is None:
            path = self.td_dir / f"wfs_{self.base}.h5"
            self._wavefunctions = read_wfs(path)
        return self._wavefunctions

    @property
    def physics(self) -> "Physics":
        if self._physics is None:
            from zkit.physics import Physics

            self._physics = Physics(self.meta)
        return self._physics

    @property
    def ts(self) -> TSSeries:
        if self._ts is None:
            path = self.run_dir / f"ts_{self.base}.h5"
            self._ts = read_timeseries(path)
        return self._ts

    # convenience aliases

    @property
    def time(self) -> "np.ndarray":
        return self.evolution.time

    @property
    def dipoles(self) -> "np.ndarray":
        return self.evolution.dipoles

    @property
    def populations(self) -> "np.ndarray":
        return self.evolution.populations

    @property
    def energies(self) -> "np.ndarray":
        return self.evolution.energies

    @property
    def currents(self) -> "np.ndarray":
        return self.evolution.currents

    @property
    def autocorrelation(self) -> "np.ndarray":
        return self.evolution.autocorrelation

    def _parse_meta(self) -> SimulationMeta:
        path = self.run_dir / self.base
        extra: Dict[str, object] = {}

        def _read(path: Path) -> Dict[str, str]:
            text = path.read_text(errors="ignore").splitlines()
            out: Dict[str, str] = {}
            for line in text:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                out[key.strip().lower()] = value.strip()
            return out

        if not path.exists():
            return SimulationMeta(input_file=str(self.input_file))

        params = _read(path)
        dimension = _to_int(params.get("dimension"), default=1)
        n_splines = _to_int(
            params.get("nsplines") or params.get("nspline") or params.get("nsplines"), default=0
        )
        time_step = _to_float(params.get("timestep"), default=0.0)
        final_time = _to_float(params.get("finaltime") or params.get("final_time"), default=0.0)
        ns_energy = _to_int(params.get("nboundstates") or params.get("nboundstatessave"), default=0)
        ns_population = _to_int(params.get("nboundstates") or params.get("npops"), default=0)
        output_stride_wfs = _to_int(params.get("outputstridewfs"), default=0)
        output_stride_ts = _to_int(params.get("outputstridets"), default=0)

        for key in (
            "dimension",
            "nsplines",
            "timestep",
            "finaltime",
            "nboundstates",
            "outputstridewfs",
            "outputstridets",
        ):
            params.pop(key, None)
        extra = params

        return SimulationMeta(
            input_file=str(self.input_file),
            dimension=dimension,
            n_splines=n_splines,
            time_step=time_step,
            final_time=final_time,
            ns_energy=ns_energy,
            ns_population=ns_population,
            output_stride_wfs=output_stride_wfs,
            output_stride_ts=output_stride_ts,
            extra=extra,
        )

    def __repr__(self) -> str:
        return (
            f"<Run run_dir={self.run_dir!r} base={self.base!r} "
            f"dim={self.meta.dimension} steps={self.evolution.n_steps}>"
        )


# Backward compatibility alias
Simulation = Run


def load(run_dir: Union[str, Path], basename: Union[str, Path]) -> Run:
    """Load a simulation run."""
    return Run(run_dir, basename)


def _to_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(str(value).split()[0])
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError):
        return default


__all__ = ["Run", "Simulation", "load"]
