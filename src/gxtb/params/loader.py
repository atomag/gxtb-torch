from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import torch
import numpy as _np

try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # pragma: no cover - fallback to tomli if available
    import tomli as _toml  # type: ignore

from .types import (
    BasisQParameters,
    D4Parameters,
    EEQElement,
    EEQParameters,
    ElementBasis,
    GxTBElementBlock,
    GxTBGlobal,
    GxTBParameters,
    Primitive,
    ShellPrimitives,
)


def _to_tensor_row(row: List[float]) -> torch.Tensor:
    return torch.tensor(row, dtype=torch.float64)


def load_gxtb_params(path: str | Path) -> GxTBParameters:
    """
    Strictly parse the `parameters/gxtb` file into a faithful, lossless structure.

    Format characteristics observed:
    - One or more numeric lines precede the first element marker; treated as global constants.
    - Element blocks start with a line that contains a single integer (atomic number Z),
      followed by one or more numeric lines of varying length. The line boundaries are
      significant and preserved.
    - All numeric lines are space-separated floating-point values.

    This loader does not assume semantic mapping of each line (that mapping will follow the
    theory equations). It guarantees exact ingestion without placeholders or silent coercions.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"gxtb parameter file not found: {p}")

    with p.open("r", encoding="utf-8") as fh:
        raw_lines = [ln.rstrip() for ln in fh]

    def is_blank(s: str) -> bool:
        return len(s.strip()) == 0

    def is_marker(s: str) -> bool:
        t = s.strip()
        return bool(re.fullmatch(r"[0-9]+", t))

    def parse_row(s: str) -> torch.Tensor:
        # Use numpy.fromstring for fast, robust float parsing
        arr = _np.fromstring(s, sep=" ", dtype=_np.float64)
        if arr.size == 0:
            raise ValueError(f"Empty or non-numeric line encountered: {s!r}")
        return torch.from_numpy(arr.copy())  # copy to detach from numpy memory

    # Identify global lines (before first marker)
    idx = 0
    n = len(raw_lines)
    global_tensors: List[torch.Tensor] = []
    while idx < n and not is_marker(raw_lines[idx]):
        if not is_blank(raw_lines[idx]):
            global_tensors.append(parse_row(raw_lines[idx]))
        idx += 1
    if idx >= n:
        raise ValueError("No element block found in gxtb parameters; missing Z markers")
    global_block = GxTBGlobal(lines=tuple(global_tensors))

    # Parse element blocks
    elements: Dict[int, GxTBElementBlock] = {}
    while idx < n:
        # Skip blanks
        if is_blank(raw_lines[idx]):
            idx += 1
            continue
        # Expect a marker
        if not is_marker(raw_lines[idx]):
            raise ValueError(f"Expected atomic number marker at line {idx+1}, got: {raw_lines[idx]!r}")
        z = int(raw_lines[idx].strip())
        idx += 1

        lines_this: List[torch.Tensor] = []
        # Collect until next marker or EOF
        while idx < n and not is_marker(raw_lines[idx]):
            if not is_blank(raw_lines[idx]):
                lines_this.append(parse_row(raw_lines[idx]))
            idx += 1

        if z in elements:
            raise ValueError(f"Duplicate entry for atomic number Z={z}")
        if len(lines_this) == 0:
            raise ValueError(f"No data lines for atomic number Z={z}")
        elements[z] = GxTBElementBlock(z=z, lines=tuple(lines_this))

    return GxTBParameters(global_lines=global_block, elements=elements)


def load_eeq_params(path: str | Path) -> EEQParameters:
    """
    Parse `parameters/eeq` into per-element 10-parameter rows.
    Each non-empty line corresponds to one atomic number (in ascending order starting at 1).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"eeq parameter file not found: {p}")
    elements: Dict[int, EEQElement] = {}
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            s = raw.strip()
            if not s:
                continue
            # Skip non-numeric lines (e.g., timestamps)
            head = s.split()[0]
            if not re.fullmatch(r"[-+0-9Ee\.]+", head):
                continue
            arr = _np.fromstring(s, sep=" ", dtype=_np.float64)
            if arr.size != 10:
                raise ValueError(f"Expected 10 values in eeq line {lineno}, got {int(arr.size)}")
            z = len(elements) + 1
            elements[z] = EEQElement(z=z, values=torch.from_numpy(arr.copy()))
    if not elements:
        raise ValueError("Empty eeq parameter file")
    return EEQParameters(elements=elements)


_re_header = re.compile(r"^\s*(\d+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s+([-+0-9Ee\.]+)\s*$")
_re_shell = re.compile(r"^\s*(\d+)\s+([spdfSPDF])\s*$")


def load_basisq(path: str | Path) -> BasisQParameters:
    """
    Parse `parameters/basisq` (q‑vSZP basis) into a structured representation.

    The format uses `*` separators between element sections. Each element section begins
    with a header line of the form:
        `<Z>  <f1>  <f2>  <f3>`
    followed by one or more shell blocks:
        `<N> <shell>`  then N lines of three floats per primitive (alpha, c1, c2).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"basisq parameter file not found: {p}")

    with p.open("r", encoding="utf-8") as fh:
        lines = [ln.rstrip() for ln in fh]

    idx = 0
    n = len(lines)
    elements: Dict[int, ElementBasis] = {}

    def skip_separators() -> None:
        nonlocal idx
        while idx < n and lines[idx].strip().startswith("*"):
            idx += 1

    while idx < n:
        skip_separators()
        if idx >= n:
            break
        # Expect header
        m = _re_header.match(lines[idx])
        if not m:
            raise ValueError(f"Expected element header at line {idx+1}, got: {lines[idx]!r}")
        z = int(m.group(1))
        header_vals = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        idx += 1

        shells: Dict[str, Tuple[ShellPrimitives, ...]] = {}

        while idx < n:
            skip_separators()
            if idx >= n:
                break
            # If next is a new header, stop current element
            if _re_header.match(lines[idx]):
                break
            # Expect a shell declaration
            m2 = _re_shell.match(lines[idx])
            if not m2:
                raise ValueError(f"Expected shell line at {idx+1}, got: {lines[idx]!r}")
            nprims = int(m2.group(1))
            shell = m2.group(2).lower()
            idx += 1
            prims: List[Primitive] = []
            for k in range(nprims):
                if idx >= n:
                    raise ValueError(f"Unexpected EOF reading primitives for Z={z} shell={shell}")
                arr = _np.fromstring(lines[idx], sep=" ", dtype=_np.float64)
                if arr.size != 3:
                    raise ValueError(
                        f"Expected 3 values in primitive at line {idx+1} for Z={z} shell={shell}, got {int(arr.size)}"
                    )
                alpha, c1, c2 = (float(arr[0]), float(arr[1]), float(arr[2]))
                prims.append(Primitive(alpha=alpha, c1=c1, c2=c2))
                idx += 1
            block = ShellPrimitives(nprims=nprims, primitives=tuple(prims))
            shells.setdefault(shell, tuple())
            shells[shell] = shells[shell] + (block,)

        if z in elements:
            raise ValueError(f"Duplicate basis entry for Z={z}")
        elements[z] = ElementBasis(z=z, header=header_vals, shells=shells)

    if not elements:
        raise ValueError("Empty basisq file")

    return BasisQParameters(elements=elements)


def load_d4_parameters(path: str | Path) -> D4Parameters:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"D4 TOML not found: {p}")
    with p.open("rb") as fh:
        data = _toml.load(fh)
    if not isinstance(data, dict) or not data:
        raise ValueError("Malformed or empty D4 TOML")
    return D4Parameters(raw=data)


def load_d4_reference_toml(path: str | Path, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> dict:
    """Load D4 reference dataset from a TOML file.

    Schema (compact, element-centric):
    - version: string
    - [sec]
        scale: [S]
        alpha: [S][W]
    - [[element]] blocks, each with:
        z: int
        refsys: [R_z]  (indices into [sec] arrays; local to this file)
        refascale: [R_z]
        refscount: [R_z]
        refalpha: [R_z][W]
        refcovcn: [R_z]
        refc: [R_z] (ints)
        clsq: [R_z]
        clsh: [R_z]
        r4r2: float
        zeff: float
        gam: float

    Returns a dict with tensors broadcastable as in tad-dftd4:
    - 'secscale': (S,)
    - 'secalpha': (S, W)
    - 'refsys','refascale','refscount','refalpha','refcovcn','refc','clsq','clsh':
      padded to shape (Zmax+1, Rmax, [W]), zeros where not provided.
    - 'r4r2','zeff','gam': length Zmax+1 with zeros where not provided.
    - 'z_supported': 1D tensor of supported Z.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"D4 reference TOML not found: {p}")
    with p.open("rb") as fh:
        data = _toml.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Malformed D4 reference TOML")
    if "sec" not in data or "element" not in data:
        raise KeyError("D4 reference TOML must contain [sec] and [[element]]")
    sec = data["sec"]
    scale = sec.get("scale")
    alpha = sec.get("alpha")
    if not isinstance(scale, list) or not isinstance(alpha, list):
        raise ValueError("[sec] must contain 'scale' and 'alpha' lists")
    S = len(scale)
    if S != len(alpha):
        raise ValueError("sec.scale and sec.alpha must have same length S")
    # Determine W from first alpha row
    if S == 0 or not isinstance(alpha[0], list):
        raise ValueError("sec.alpha must be a list of W-length lists")
    W = len(alpha[0])
    for row in alpha:
        if len(row) != W:
            raise ValueError("All rows of sec.alpha must have equal length W")
    elts = data["element"]
    if not isinstance(elts, list) or not elts:
        raise ValueError("No [[element]] section found in D4 reference TOML")
    # Collect Z set and per-element R_z
    z_list: List[int] = []
    Rzs: Dict[int, int] = {}
    for blk in elts:
        z = int(blk.get("z"))
        z_list.append(z)
        Rzs[z] = int(len(blk.get("refsys", [])))
    Zmax = max(z_list)
    Rmax = max(Rzs.values()) if Rzs else 0
    to_dtype = dtype if dtype is not None else torch.float64
    # Allocate padded tensors
    def zeros_ZR(shape_last: int = 1, *, integer: bool = False) -> torch.Tensor:
        base_shape = (Zmax + 1, Rmax)
        if shape_last > 1:
            base_shape = base_shape + (shape_last,)
        dt = torch.int64 if integer else to_dtype
        return torch.zeros(base_shape, device=device, dtype=dt)
    refsys_t = zeros_ZR(integer=True)
    refascale_t = zeros_ZR()
    refscount_t = zeros_ZR()
    refalpha_t = zeros_ZR(W)
    refcovcn_t = zeros_ZR()
    refc_t = zeros_ZR(integer=True)
    clsq_t = zeros_ZR()
    clsh_t = zeros_ZR()
    r4r2_t = torch.zeros((Zmax + 1,), device=device, dtype=to_dtype)
    zeff_t = torch.zeros((Zmax + 1,), device=device, dtype=to_dtype)
    gam_t = torch.zeros((Zmax + 1,), device=device, dtype=to_dtype)
    # Fill element blocks
    for blk in elts:
        z = int(blk["z"])
        Rz = Rzs[z]
        def to_t(arr, *, integer=False):
            return torch.tensor(arr, device=device, dtype=(torch.int64 if integer else to_dtype))
        refsys = to_t(blk["refsys"], integer=True)
        refascale = to_t(blk["refascale"]) if "refascale" in blk else torch.zeros(Rz, device=device, dtype=to_dtype)
        refscount = to_t(blk["refscount"]) if "refscount" in blk else torch.zeros(Rz, device=device, dtype=to_dtype)
        refalpha = to_t(blk["refalpha"])  # (Rz, W)
        refcovcn = to_t(blk["refcovcn"]) if "refcovcn" in blk else torch.zeros(Rz, device=device, dtype=to_dtype)
        refc = to_t(blk["refc"], integer=True) if "refc" in blk else torch.zeros(Rz, device=device, dtype=torch.int64)
        clsq = to_t(blk["clsq"]) if "clsq" in blk else torch.zeros(Rz, device=device, dtype=to_dtype)
        clsh = to_t(blk["clsh"]) if "clsh" in blk else torch.zeros(Rz, device=device, dtype=to_dtype)
        # Copy into padded slots
        refsys_t[z, :Rz] = refsys
        refascale_t[z, :Rz] = refascale
        refscount_t[z, :Rz] = refscount
        refalpha_t[z, :Rz, :] = refalpha
        refcovcn_t[z, :Rz] = refcovcn
        refc_t[z, :Rz] = refc
        clsq_t[z, :Rz] = clsq
        clsh_t[z, :Rz] = clsh
        r4r2_t[z] = float(blk.get("r4r2", 0.0))
        zeff_t[z] = float(blk.get("zeff", 0.0))
        gam_t[z] = float(blk.get("gam", 0.0))
    # sec arrays
    secscale_t = torch.tensor(scale, device=device, dtype=to_dtype)
    secalpha_t = torch.tensor(alpha, device=device, dtype=to_dtype)
    return {
        'secscale': secscale_t,
        'secalpha': secalpha_t,
        'refsys': refsys_t,
        'refascale': refascale_t,
        'refscount': refscount_t,
        'refalpha': refalpha_t,
        'refcovcn': refcovcn_t,
        'refc': refc_t,
        'clsq': clsq_t,
        'clsh': clsh_t,
        'r4r2': r4r2_t,
        'zeff': zeff_t,
        'gam': gam_t,
        'z_supported': torch.tensor(sorted(set(z_list)), device=device, dtype=torch.int64),
    }


def select_d4_params(
    d4: D4Parameters | dict,
    *,
    method: str = "d4",
    functional: str | None,
    variant: str | None = None,
    keep_doi: bool = False,
) -> dict:
    """
    Select a D4 damping parameter record from a TOML table, matching the
    semantics of tad-dftd4 (ref: tad_dftd4/damping/parameters/loader.py).

    Inputs
    - d4: either D4Parameters or raw TOML dict (as from load_d4_parameters)
    - method: dispersion family (only 'd4' supported here)
    - functional: DFT functional name (case-insensitive) or None/'default'
    - variant: damping variant key, e.g., 'bj-eeq-atm'; if None, use default
      variant from [default].[method] list in the TOML.

    Returns a shallow copy dict with scalar fields like s6, s8, s9, a1, a2,
    alp, damping, mbd, doi (if keep_doi=True).

    Raises KeyError on missing functional/method/variant entries.
    """
    table = d4.raw if isinstance(d4, D4Parameters) else d4
    if not isinstance(table, dict):  # defensive
        raise TypeError("select_d4_params expects D4Parameters or raw dict")

    method_key = method.lower()
    if method_key != "d4":
        raise KeyError(f"Unsupported dispersion method: {method}")

    default_section = table.get("default")
    if default_section is None:
        raise KeyError("Missing [default] section in D4 TOML")
    default_variants = default_section.get(method_key)
    if not isinstance(default_variants, list) or not default_variants:
        raise KeyError(f"Missing default variants list for method={method_key}")

    if functional in (None, "default"):
        disp_method_section = default_section.get("parameter")
        if disp_method_section is None:
            raise KeyError("Missing [default.parameter] section in D4 TOML")
    else:
        func_section = table.get("parameter", {})
        fkey = str(functional).casefold()
        if fkey not in func_section:
            raise KeyError(f"Functional '{functional}' not found in D4 parameters")
        disp_method_section = func_section[fkey]

    if method_key not in disp_method_section:
        raise KeyError(f"Method '{method_key}' not found in selected parameter section")
    variant_section = disp_method_section[method_key]

    if variant is None:
        variant = default_variants[0]
    if variant not in variant_section:
        raise KeyError(
            f"Variant '{variant}' not found for functional={functional!r}, method={method_key!r}"
        )

    block = dict(variant_section[variant])  # shallow copy
    if not keep_doi and "doi" in block:
        block.pop("doi", None)
    return block


# NPZ-based loaders removed: D4 reference data is loaded from a TOML file
# (see load_d4_reference_toml) or provided explicitly by the caller.


def validate_parameter_coverage(param_dir: str | Path) -> dict:
    """Cross-check element coverage across gxtb, eeq, basisq parameter files.

    Ensures:
    - Each file exists and loads.
    - Non-empty element sets.
    - Intersection of Z sets non-empty.
    Returns a dict of sets for reporting.
    """
    p = Path(param_dir)
    g = load_gxtb_params(p / 'gxtb')
    e = load_eeq_params(p / 'eeq')
    b = load_basisq(p / 'basisq')
    gZ = set(g.elements.keys())
    eZ = set(e.elements.keys())
    bZ = set(b.elements.keys())
    if not gZ or not eZ or not bZ:
        raise ValueError("One or more parameter files have empty element sets (gxtb/eeq/basisq)")
    inter = gZ & eZ & bZ
    if not inter:
        raise ValueError("No common elements across gxtb, eeq, and basisq parameter sets")
    return {'gxtb': gZ, 'eeq': eZ, 'basisq': bZ, 'common': inter}
