"""Read TDSE-Z input decks with the rules of the TDSE-Z parser.

The reader mirrors ``src/parser.cpp`` of TDSE-Z:

* ``#`` starts a comment anywhere on a line;
* ``Key = value`` splits at the first ``=``;
* key aliases (``NSplines`` -> ``Nelements``, ``DriverX`` -> ``LaserX``, ...)
  resolve to one canonical key; keys are matched case-insensitively here;
* ``Domain = lo, hi`` or ``Domain = [lo,hi],[lo,hi],[lo,hi]`` sets the per-axis
  bounds (missing axes copy x);
* the per-axis list keys (``LaserAmplitude = a, b, c``, ...) fill
  Ampx/Ampy/Ampz etc.; a single value is used for every axis;
* ``Variables = Name=value, ...`` lines accumulate user constants;
* every input scalar TDSE-Z binds inside expressions is available, with the
  TDSE-Z default when the deck does not set it (:func:`expression_constants`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

__all__ = ["read_deck", "expression_constants", "ALIASES"]

# alias (lower case) -> canonical key (lower case)
ALIASES: Dict[str, str] = {
    "savestateformat": "boundstateformat",
    "stateformat": "boundstateformat",
    "boundary": "boundarytype",
    "absorbingboundary": "enablecap",
    "lzdiag": "enablelzdiag",
    "totaltime": "finaltime",
    "startstate": "initialstate",
    "knotgrading": "knotalpha",
    "gradingstrength": "knotalpha",
    "driver": "laser",
    "driverx": "laserx",
    "drivery": "lasery",
    "driverz": "laserz",
    "numberofboundstates": "nboundstates",
    "saveboundstates": "nboundstatessave",
    "nsplines": "nelements",
    "numberofsplines": "nelements",
    "elementcount": "nelements",
    "numberofelements": "nelements",
    "normalizepsi0": "normalizeinitialstate",
    "outputquantities": "physicsoutput",
    "laserpolarization": "polarization",
    "driverpolarization": "polarization",
    "externalpotential": "potential",
    "particlecharge": "charge",
    "targetenergy": "targeteigenvalue",
    "timestepsize": "timestep",
    "directsolve": "usedirectsolve",
}

# per-axis list keys -> the three canonical scalar keys they fill
_LIST_KEYS: Dict[str, Tuple[str, str, str]] = {
    "laseramplitude": ("ampx", "ampy", "ampz"),
    "driveramplitude": ("ampx", "ampy", "ampz"),
    "laserfrequency": ("omegax", "omegay", "omegaz"),
    "driverfrequency": ("omegax", "omegay", "omegaz"),
    "carrierenvelopephase": ("cepx", "cepy", "cepz"),
    "drivercep": ("cepx", "cepy", "cepz"),
}

# Scalars TDSE-Z binds as constants in every expression: name -> (canonical
# key, TDSE-Z default). Names are case-sensitive, as in muParser.
_CONSTANTS: Dict[str, Tuple[str, float]] = {
    "Q": ("charge", 1.0),
    "Charge": ("charge", 1.0),
    "Dimension": ("dimension", 1.0),
    "SplineDegree": ("splinedegree", 3.0),
    "Nelements": ("nelements", 100.0),
    "NSplines": ("nelements", 100.0),
    "LMinX": ("lminx", -10.0),
    "LMaxX": ("lmaxx", 10.0),
    "LMinY": ("lminy", -10.0),
    "LMaxY": ("lmaxy", 10.0),
    "LMinZ": ("lminz", -10.0),
    "LMaxZ": ("lmaxz", 10.0),
    "OffsetX": ("offsetx", 0.0),
    "OffsetY": ("offsety", 0.0),
    "OffsetZ": ("offsetz", 0.0),
    "Ampx": ("ampx", 0.0),
    "Ampy": ("ampy", 0.0),
    "Ampz": ("ampz", 0.0),
    "CEPx": ("cepx", 0.0),
    "CEPy": ("cepy", 0.0),
    "CEPz": ("cepz", 0.0),
    "Omegax": ("omegax", 0.0),
    "Omegay": ("omegay", 0.0),
    "Omegaz": ("omegaz", 0.0),
    "TargetEigenvalue": ("targeteigenvalue", 0.0),
    "TimeStep": ("timestep", 0.01),
    "FinalTime": ("finaltime", 100.0),
    "CAPKmin": ("capkmin", 0.1),
    "KnotAlpha": ("knotalpha", 0.0),
    "NQuadratures": ("nquadratures", 8.0),
    "NBoundStates": ("nboundstates", 1.0),
}


def _split_list(value: str):
    return [tok for tok in value.replace(" ", "").split(",") if tok != ""]


def _parse_domain(value: str, params: Dict[str, str]) -> None:
    s = value.replace(" ", "")
    axes = "xyz"
    if "[" in s:
        pairs = []
        pos = 0
        while True:
            ob = s.find("[", pos)
            cb = s.find("]", ob + 1) if ob >= 0 else -1
            if ob < 0 or cb < 0:
                break
            pairs.append(s[ob + 1 : cb])
            pos = cb + 1
        for i, ax in enumerate(axes):
            pair = pairs[i] if i < len(pairs) else (pairs[0] if pairs else None)
            if pair and "," in pair:
                lo, hi = pair.split(",", 1)
                params["lmin" + ax], params["lmax" + ax] = lo, hi
    elif "," in s:
        lo, hi = s.split(",", 1)
        for ax in axes:
            params["lmin" + ax], params["lmax" + ax] = lo, hi


def read_deck(path: Union[str, Path]) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Read a TDSE-Z input deck.

    Returns
    -------
    params:
        canonical lower-case key -> raw value string (last occurrence wins;
        TDSE-Z itself rejects a key given twice).
    variables:
        user constants from all ``Variables`` lines, name -> value.
    """
    params: Dict[str, str] = {}
    variables: Dict[str, float] = {}
    for raw in Path(path).read_text(errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().lower(), value.strip()
        key = ALIASES.get(key, key)
        if key == "variables":
            for pair in _split_list(value):
                name, eq, num = pair.partition("=")
                if eq and name:
                    variables[name] = float(num)
        elif key == "domain":
            _parse_domain(value, params)
        elif key in _LIST_KEYS:
            vals = _split_list(value)
            if vals:
                for i, target in enumerate(_LIST_KEYS[key]):
                    params[target] = vals[i] if i < len(vals) else vals[0]
        else:
            params[key] = value
    return params, variables


def expression_constants(params: Dict[str, str], variables: Dict[str, float]) -> Dict[str, float]:
    """Constants available inside TDSE-Z expressions: ``pi``, every bound input
    scalar (deck value or TDSE-Z default) and the user ``Variables``."""
    import math

    out: Dict[str, float] = {"pi": math.pi, "_pi": math.pi, "_e": math.e}
    for name, (key, default) in _CONSTANTS.items():
        try:
            out[name] = float(str(params[key]).split()[0]) if key in params else default
        except ValueError:
            out[name] = default
    out.update(variables)
    return out
