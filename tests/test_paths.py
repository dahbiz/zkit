"""Output names of a run: static/<name>_*, td/<name>_*, and the former names."""

from pathlib import Path

import pytest

from zkit import Run
from zkit.paths import list_runs, output_path, run_name


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def test_run_name_strips_directory_and_extension() -> None:
    assert run_name("runs/h2p.inp") == "h2p"
    assert run_name("h2p") == "h2p"
    assert run_name("ZnSe_d20_A0.005338.inp") == "ZnSe_d20_A0.005338"


@pytest.mark.parametrize(
    "kind,new",
    [
        ("eigen", "static/h2p_eigen.h5"),
        ("tdm", "static/h2p_tdm.txt"),
        ("observables", "td/h2p_observables.h5"),
        ("wfs", "td/h2p_wfs.h5"),
        ("surff", "td/h2p_surff.h5"),
        ("ts", "td/h2p_ts.h5"),
    ],
)
def test_current_names(tmp_path, kind, new) -> None:
    assert output_path(tmp_path, "h2p.inp", kind) == tmp_path / new  # not written yet
    _touch(tmp_path / new)
    assert output_path(tmp_path, "h2p.inp", kind) == tmp_path / new
    assert Run(tmp_path, "h2p.inp").path(kind) == tmp_path / new


@pytest.mark.parametrize(
    "kind,old",
    [
        ("eigen", "static/EigenData_h2p.inp.h5"),
        ("tdm", "static/TDMInfo_h2p.inp.txt"),
        ("observables", "td/TimeEvolutionData_h2p.inp.h5"),
        ("wfs", "td/wfs_h2p.inp.h5"),
        ("surff", "td/surff_h2p.inp.h5"),
        ("ts", "ts_h2p.inp.h5"),
    ],
)
def test_former_names_are_found(tmp_path, kind, old) -> None:
    _touch(tmp_path / old)
    assert output_path(tmp_path, "h2p.inp", kind) == tmp_path / old
    assert output_path(tmp_path, "h2p", kind) == tmp_path / old


def test_list_runs_covers_both_layouts(tmp_path) -> None:
    _touch(tmp_path / "td" / "a_observables.h5")
    _touch(tmp_path / "td" / "TimeEvolutionData_b.inp.h5")
    assert list_runs(tmp_path) == ["a", "b"]
