"""Dimension-specific column layouts for TDSEZ time-evolution datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DipoleLayout:
    """Column layout for the dipoles dataset."""

    time: int = 0
    laser_ex: int = 1
    laser_ey: int = -1
    laser_ez: int = -1
    dipole_x: int = -1
    dipole_y: int = -1
    dipole_z: int = -1
    accel_x: int = -1
    accel_y: int = -1
    accel_z: int = -1

    @property
    def width(self) -> int:
        return (
            max(
                self.time,
                self.laser_ex,
                self.laser_ey,
                self.laser_ez,
                self.dipole_x,
                self.dipole_y,
                self.dipole_z,
                self.accel_x,
                self.accel_y,
                self.accel_z,
            )
            + 1
        )

    def asdict(self) -> dict[str, int | None]:
        d: dict[str, int | None] = {
            "time": self.time,
            "laser_ex": self.laser_ex,
            "dipole_x": self.dipole_x,
            "accel_x": self.accel_x,
        }
        if self.laser_ey >= 0:
            d["laser_ey"] = self.laser_ey
        if self.laser_ez >= 0:
            d["laser_ez"] = self.laser_ez
        if self.dipole_y >= 0:
            d["dipole_y"] = self.dipole_y
        if self.dipole_z >= 0:
            d["dipole_z"] = self.dipole_z
        if self.accel_y >= 0:
            d["accel_y"] = self.accel_y
        if self.accel_z >= 0:
            d["accel_z"] = self.accel_z
        return d


_1D_DIPOLE = DipoleLayout(
    time=0,
    laser_ex=1,
    dipole_x=2,
    accel_x=3,
)

_2D_DIPOLE = DipoleLayout(
    time=0,
    laser_ex=1,
    laser_ey=2,
    dipole_x=3,
    dipole_y=4,
    accel_x=5,
    accel_y=6,
)

_3D_DIPOLE = DipoleLayout(
    time=0,
    laser_ex=1,
    laser_ey=2,
    laser_ez=3,
    dipole_x=4,
    dipole_y=5,
    dipole_z=6,
    accel_x=7,
    accel_y=8,
    accel_z=9,
)

ENERGY_COLUMNS = ("time", "kinetic", "potential", "interaction", "total", "inv_mass_avg", "norm_sq")
CURRENT_COLUMNS = (
    "time",
    "lz",
    "d_gamma",
    "jx_total",
    "jy_total",
    "jx_intra",
    "jy_intra",
    "jx_inter",
    "jy_inter",
    "jx_bc",
    "jy_bc",
)
# 3D layout adds the z-current components (Jz_total/intra/inter/bc) -> width 15.
CURRENT_COLUMNS_3D = (
    "time",
    "lz",
    "d_gamma",
    "jx_total",
    "jy_total",
    "jz_total",
    "jx_intra",
    "jy_intra",
    "jz_intra",
    "jx_inter",
    "jy_inter",
    "jz_inter",
    "jx_bc",
    "jy_bc",
    "jz_bc",
)
AC_COLUMNS = ("time", "re_autocorr", "im_autocorr")


def current_columns(dim: int) -> tuple:
    """Column layout for the `currents` dataset for the given dimension.

    1D/2D store an (x, y) decomposition (width 11); 3D adds the z components
    (width 15)."""
    if dim == 3:
        return CURRENT_COLUMNS_3D
    return CURRENT_COLUMNS


def dipole_layout(dim: int) -> DipoleLayout:
    if dim == 1:
        return _1D_DIPOLE
    if dim == 2:
        return _2D_DIPOLE
    if dim == 3:
        return _3D_DIPOLE
    raise ValueError(f"Unsupported dimension: {dim!r}. Expected 1, 2, or 3.")


__all__ = [
    "DipoleLayout",
    "dipole_layout",
    "ENERGY_COLUMNS",
    "CURRENT_COLUMNS",
    "CURRENT_COLUMNS_3D",
    "current_columns",
    "AC_COLUMNS",
]
