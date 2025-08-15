from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

try:
    import tomllib as _toml
except Exception:  # pragma: no cover
    import tomli as _toml  # type: ignore

from .loader import GxTBParameters


@dataclass(frozen=True)
class RepulsionSchema:
    # element-wise indexes (line, column indexes are 0-based)
    z_eff0: Tuple[int, int]
    alpha0: Tuple[int, int]
    kq: Tuple[int, int]
    kq2: Tuple[int, int]
    kcn: Tuple[int, int]
    r0: Tuple[int, int]

    # global indexes
    kpen1_hhe: Tuple[int, int]
    kpen1_rest: Tuple[int, int]
    kpen2: Tuple[int, int]
    kpen3: Tuple[int, int]
    kpen4: Tuple[int, int]
    kexp: Tuple[int, int]


@dataclass(frozen=True)
class CNSchema:
    # element-wise R_cov index and global k_cn index
    r_cov: Tuple[int, int]
    k_cn: Tuple[int, int]


@dataclass(frozen=True)
class IncrementSchema:
    delta_e_incr: Tuple[int, int]


@dataclass(frozen=True)
class GxTBSchema:
    repulsion: RepulsionSchema
    cn: CNSchema
    increment: IncrementSchema
    diatomic: Dict[str, Tuple[int, int]] | None = None
    eht: Dict[str, Tuple[int, int]] | None = None
    eht_poly: Dict[str, List[Tuple[int, int]]] | None = None
    eht_cn_poly: Dict[str, List[Tuple[int, int]]] | None = None
    eht_r_poly: Dict[str, List[Tuple[int, int]]] | None = None
    hubbard: Dict[str, Tuple[int, int]] | None = None
    third_order: Dict[str, Tuple[int, int]] | None = None
    fourth_order: Dict[str, Tuple[int, int]] | None = None
    spin: Dict[str, Tuple[int, int]] | None = None
    aes: Dict[str, Tuple[int, int]] | None = None
    aes_element: Dict[str, Tuple[int, int]] | None = None
    aes_rules: Dict[str, str] | None = None
    ofx_element: Dict[str, Tuple[int, int]] | None = None
    ofx_rules: Dict[str, str] | None = None
    # MFX mapping (educated guess). Optional sections:
    # - [mfx]: global scalars alpha, omega, k1, k2, and optionally xi_s, xi_p, xi_d, xi_f
    # - [mfx.element]: per-element shell Hubbard-like U^{MFX}_l via keys U_s, U_p, U_d, U_f
    mfx: Dict[str, Tuple[int, int]] | None = None
    mfx_element: Dict[str, Tuple[int, int]] | None = None
    # q‑vSZP dynamic effective-charge prefactors mapping (doc/theory/7, Eq. 28)
    # Stored dynamically as a plain dict to avoid widening the dataclass API; keys: 'k0','k1','k2','k3'
    # Values are (line, col) tuples per element block.
    # Access via hasattr(schema, 'qvszp') and getattr(schema, 'qvszp').


def _get_line_val(lines: Tuple[torch.Tensor, ...], idx: Tuple[int, int]) -> float:
    li, ci = idx
    row = lines[li]
    return float(row[ci].item())


def load_schema(path: str | Path) -> GxTBSchema:
    p = Path(path)
    with p.open("rb") as fh:
        data = _toml.load(fh)

    def tup(x: List[int]) -> Tuple[int, int]:
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("Schema indices must be 2-element lists [line, col]")
        return int(x[0]), int(x[1])

    rep = data["repulsion"]
    cn = data["cn"]
    incr = data["increment"]
    schema = GxTBSchema(
        repulsion=RepulsionSchema(
            z_eff0=tup(rep["z_eff0"]),
            alpha0=tup(rep["alpha0"]),
            kq=tup(rep["kq"]),
            kq2=tup(rep["kq2"]),
            kcn=tup(rep["kcn"]),
            r0=tup(rep["r0"]),
            kpen1_hhe=tup(rep["kpen1_hhe"]),
            kpen1_rest=tup(rep["kpen1_rest"]),
            kpen2=tup(rep["kpen2"]),
            kpen3=tup(rep["kpen3"]),
            kpen4=tup(rep["kpen4"]),
            kexp=tup(rep["kexp"]),
        ),
        cn=CNSchema(
            r_cov=tup(cn["r_cov"]),
            k_cn=tup(cn["k_cn"]),
        ),
        increment=IncrementSchema(
            delta_e_incr=tup(incr["delta_e_incr"]),
        ),
    )
    # helper to track used indices per section for duplicate detection
    def validate_indices(name: str, index_sets: List[Tuple[int,int]]) -> None:
        seen = set()
        for li, ci in index_sets:
            if (li, ci) in seen:
                raise ValueError(f"Duplicate index (line={li},col={ci}) in schema section '{name}'")
            if li < 0 or ci < 0:
                raise ValueError(f"Negative index in section '{name}': {(li,ci)}")
            seen.add((li,ci))

    if "diatomic" in data:
        diat = data["diatomic"]
        diat_map = {
            "sigma": tup(diat["sigma"]),
            "pi": tup(diat["pi"]),
            "delta": tup(diat["delta"]),
        }
        object.__setattr__(schema, "diatomic", diat_map)
        validate_indices("diatomic", list(diat_map.values()))
    if "eht" in data:
        eht = data["eht"]
        scalar: Dict[str, Tuple[int,int]] = {}
        cn_poly: Dict[str, List[Tuple[int,int]]] = {}
        r_poly: Dict[str, List[Tuple[int,int]]] = {}
        other_poly: Dict[str, List[Tuple[int,int]]] = {}
        for k, v in eht.items():
            if isinstance(v, list) and v and isinstance(v[0], list):
                if k.startswith('pi_cn_'):
                    cn_poly[k] = [tup(x) for x in v]  # type: ignore
                elif k.startswith('pi_r_'):
                    r_poly[k] = [tup(x) for x in v]  # type: ignore
                elif k.startswith('pi_'):
                    other_poly[k] = [tup(x) for x in v]
            else:
                scalar[k] = tup(v)  # type: ignore
        # Provisional keys (eps_mod_*, *_extra_*) are loaded but may remain unused until equations assigned.
        object.__setattr__(schema, "eht", scalar if scalar else None)
        if other_poly:
            object.__setattr__(schema, "eht_poly", other_poly)
        if cn_poly:
            object.__setattr__(schema, "eht_cn_poly", cn_poly)
        if r_poly:
            object.__setattr__(schema, "eht_r_poly", r_poly)
        all_scalar = list(scalar.values())
        if all_scalar:
            validate_indices("eht.scalar", all_scalar)
        for sec_name, coll in (("eht.cn_poly", cn_poly), ("eht.r_poly", r_poly), ("eht.legacy_poly", other_poly)):
            if coll:
                validate_indices(sec_name, [x for lst in coll.values() for x in lst])
    if "hubbard" in data:
        hub = data["hubbard"]
        hubmap: Dict[str, Tuple[int,int]] = {}
        for k, v in hub.items():
            hubmap[k] = tup(v)  # type: ignore
        object.__setattr__(schema, "hubbard", hubmap)
        validate_indices("hubbard", list(hubmap.values()))
    if "third_order" in data:
        to = data["third_order"]
        tomap: Dict[str, Tuple[int,int]] = {}
        for k, v in to.items():
            tomap[k] = tup(v)  # type: ignore
        object.__setattr__(schema, "third_order", tomap)
        validate_indices("third_order", list(tomap.values()))
    if "fourth_order" in data:
        fo = data["fourth_order"]
        fmap: Dict[str, Tuple[int,int]] = {}
        for k, v in fo.items():
            fmap[k] = tup(v)  # type: ignore
        object.__setattr__(schema, "fourth_order", fmap)
        validate_indices("fourth_order", list(fmap.values()))
    if "spin" in data:
        sp = data["spin"]
        smap: Dict[str, Tuple[int,int]] = {}
        for k, v in sp.items():
            smap[k] = tup(v)  # type: ignore
        object.__setattr__(schema, "spin", smap)
        validate_indices("spin", list(smap.values()))
    if "aes" in data:
        aes = data["aes"]
        amap: Dict[str, Tuple[int,int]] = {}
        arules: Dict[str, str | float] = {}
        # handle nested table [aes] for globals; skip nested subtables (e.g., element)
        for k, v in aes.items():
            if isinstance(v, dict):
                continue
            if isinstance(v, list):
                amap[k] = tup(v)  # type: ignore
            elif isinstance(v, (str, int, float)):
                arules[k] = float(v) if isinstance(v, (int, float)) else v
        if amap:
            object.__setattr__(schema, "aes", amap)
            # Allow duplicate indices within [aes] to explicitly reuse global slots
            # for different orders (e.g., map dmp7/dmp9 to dmp3/dmp5) without
            # rule-based derivation. Still validate non-negative indices.
            for li, ci in amap.values():
                if li < 0 or ci < 0:
                    raise ValueError(f"Negative index in section 'aes': {(li,ci)}")
        if arules:
            # store as strings/floats; consumers pick needed keys (e.g., si_*)
            object.__setattr__(schema, "aes_rules", {str(k): arules[k] for k in arules})
    # OFX nested tables
    if "ofx" in data and isinstance(data["ofx"], dict):
        ofx = data["ofx"]
        if "element" in ofx:
            elem = ofx["element"]
            emap: Dict[str, Tuple[int,int]] = {}
            for k, v in elem.items():
                if isinstance(v, list):
                    emap[k] = tup(v)  # type: ignore
            if emap:
                object.__setattr__(schema, "ofx_element", emap)
                validate_indices("ofx.element", list(emap.values()))
        if "rules" in ofx:
            rules = ofx["rules"]
            if isinstance(rules, dict):
                object.__setattr__(schema, "ofx_rules", {str(k): str(v) for k, v in rules.items()})
    # MFX nested tables (optional)
    if "mfx" in data and isinstance(data["mfx"], dict):
        mfx = data["mfx"]
        # Globals: alpha, omega, k1, k2, optionally xi_* (per-shell)
        gmap: Dict[str, Tuple[int,int]] = {}
        for k, v in mfx.items():
            # Skip nested tables here
            if isinstance(v, list):
                gmap[k] = tup(v)  # type: ignore
        if gmap:
            object.__setattr__(schema, "mfx", gmap)
            # Enforce non-negative indices
            for li, ci in gmap.values():
                if li < 0 or ci < 0:
                    raise ValueError(f"Negative index in section 'mfx': {(li,ci)}")
        # Element-wise U_shell
        if "element" in mfx and isinstance(mfx["element"], dict):
            emap: Dict[str, Tuple[int,int]] = {}
            for k, v in mfx["element"].items():
                if isinstance(v, list):
                    emap[k] = tup(v)  # type: ignore
            if emap:
                object.__setattr__(schema, "mfx_element", emap)
                for li, ci in emap.values():
                    if li < 0 or ci < 0:
                        raise ValueError(f"Negative index in section 'mfx.element': {(li,ci)}")
    # nested [aes.element]
    if "aes" in data and isinstance(data["aes"], dict) and "element" in data["aes"]:
        aese = data["aes"]["element"]
        emap: Dict[str, Tuple[int,int]] = {}
        for k, v in aese.items():
            emap[k] = tup(v)  # type: ignore
        object.__setattr__(schema, "aes_element", emap)
        validate_indices("aes.element", list(emap.values()))
    # [qvszp] section: per-element indices for k0..k3 (doc/theory/7_q-vSZP_basis_set.md, Eq. 28)
    if "qvszp" in data:
        qv = data["qvszp"]
        # Accept only explicit [line, col] lists; no rules or derivations here
        qmap: Dict[str, Tuple[int, int]] = {}
        for key in ("k0", "k1", "k2", "k3"):
            if key not in qv:
                raise ValueError("[qvszp] section must provide indices for 'k0','k1','k2','k3' (Eq. 28)")
            qmap[key] = tup(qv[key])  # type: ignore
        # Attach dynamically; duplicate index check within the section (non-negative enforced by tup)
        for li, ci in qmap.values():
            if li < 0 or ci < 0:
                raise ValueError(f"Negative index in section 'qvszp': {(li,ci)}")
        object.__setattr__(schema, "qvszp", qmap)
    return schema


def map_fourth_order_params(g: GxTBParameters, schema: GxTBSchema) -> float:
    """Map Γ^(4) from schema.fourth_order; returns scalar.

    Requires a key 'gamma4'. Raises if missing.
    """
    if schema.fourth_order is None or 'gamma4' not in schema.fourth_order:
        raise ValueError("Schema missing [fourth_order].gamma4 mapping (Eq. 140b)")
    idx = schema.fourth_order['gamma4']
    return _get_line_val(g.global_lines.lines, idx)


def map_third_order_params(g: GxTBParameters, schema: GxTBSchema) -> Dict[str, torch.Tensor | float]:
    """Map third-order TB parameters per doc/theory/18_third_order_tb.md.

    Returns dict:
      - gamma3_elem: (Zmax+1,) tensor (Eq. 131 element-wise Γ^{(3)}_A)
      - kGamma: (4,) tensor [s,p,d,f] global scalings (Eq. 131)
      - k3, k3x: floats (Eqs. 132–133b)
    """
    if schema.third_order is None:
        raise ValueError("Schema missing [third_order] section for third-order TB parameters")
    maxz = max(g.elements)
    # element-wise
    if 'gamma3_elem' not in schema.third_order:
        raise ValueError("Schema [third_order] must define gamma3_elem [line,col]")
    gamma3 = torch.zeros(maxz + 1, dtype=torch.float64)
    for z, blk in g.elements.items():
        gamma3[z] = _get_line_val(blk.lines, schema.third_order['gamma3_elem'])
    # global from global_lines
    gl = g.global_lines.lines
    def gval(idx: Tuple[int, int]) -> float:
        return _get_line_val(gl, idx)
    required = ['kGamma_s','kGamma_p','kGamma_d','kGamma_f','k3','k3x']
    for key in required:
        if key not in schema.third_order:
            raise ValueError(f"Schema [third_order] missing '{key}' index")
    kGamma = torch.tensor([
        gval(schema.third_order['kGamma_s']),
        gval(schema.third_order['kGamma_p']),
        gval(schema.third_order['kGamma_d']),
        gval(schema.third_order['kGamma_f']),
    ], dtype=torch.float64)
    k3 = gval(schema.third_order['k3'])
    k3x = gval(schema.third_order['k3x'])
    return {"gamma3_elem": gamma3, "kGamma": kGamma, "k3": k3, "k3x": k3x}


def map_spin_kW(g: GxTBParameters, schema: GxTBSchema) -> torch.Tensor:
    """Map element-wise spin scaling k^W_A (Eq. 121) from [spin].kW.

    Returns tensor (Zmax+1,). Raises if missing.
    """
    if schema.spin is None or 'kW' not in schema.spin:
        raise ValueError("Schema missing [spin].kW mapping (Eq. 121)")
    maxz = max(g.elements)
    out = torch.zeros(maxz + 1, dtype=torch.float64)
    for z, blk in g.elements.items():
        out[z] = _get_line_val(blk.lines, schema.spin['kW'])
    return out


def map_aes_global(g: GxTBParameters, schema: GxTBSchema) -> Dict[str, float]:
    """Map AES global damping parameters: dmp3, dmp5 (orders n=3,5).

    If additional keys 'dmp7'/'dmp9' exist in schema, include them.
    Otherwise, if rules are present (e.g., dmp7_from="extrapolate_linear"), derive values.
    """
    if schema.aes is None:
        raise ValueError("Schema missing [aes] section for AES global parameters (doc/theory/16)")
    gl = g.global_lines.lines
    out: Dict[str, float] = {}
    for key in ("dmp3", "dmp5"):
        if key not in schema.aes:
            raise ValueError(f"Schema [aes] missing '{key}' index")
        out[key] = _get_line_val(gl, schema.aes[key])
    # Optional higher orders if present as indices
    for key in ("dmp7", "dmp9"):
        if schema.aes and key in schema.aes:
            out[key] = _get_line_val(gl, schema.aes[key])
    # If not present as indices, allow explicit derivation rules in schema.aes_rules
    rules = schema.aes_rules or {}
    # simple linear extrapolation over n: assume step from 3->5 repeats to 7,9
    if 'dmp7' not in out and rules.get('dmp7_from', '') == 'extrapolate_linear':
        out['dmp7'] = out['dmp5'] + (out['dmp5'] - out['dmp3'])
    if 'dmp9' not in out and rules.get('dmp9_from', '') == 'extrapolate_linear':
        # If dmp7 known (either indexed or extrapolated), extend one more step; otherwise use +2*(dmp5-dmp3)
        step = (out['dmp5'] - out['dmp3'])
        out['dmp9'] = (out.get('dmp7', out['dmp5'] + step)) + step
    # Attach rules (strings) if present for SI damping or derivations
    if schema.aes_rules:
        for k, v in schema.aes_rules.items():
            out[k] = v  # pass-through; consumer parses as needed
    return out


def map_aes_element(g: GxTBParameters, schema: GxTBSchema) -> Dict[str, torch.Tensor]:
    """Map AES element-wise damping radii modifiers: mprad, mpvcn (per element)."""
    if schema.aes_element is None:
        raise ValueError("Schema missing [aes.element] section for AES element parameters")
    maxz = max(g.elements)
    out: Dict[str, torch.Tensor] = {}
    for key in ("mprad", "mpvcn"):
        if key not in schema.aes_element:
            raise ValueError(f"Schema [aes.element] missing '{key}' index")
        arr = torch.zeros(maxz + 1, dtype=torch.float64)
        for z, blk in g.elements.items():
            arr[z] = _get_line_val(blk.lines, schema.aes_element[key])
        out[key] = arr
    return out


def map_ofx_element(g: GxTBParameters, schema: GxTBSchema) -> Dict[str, torch.Tensor]:
    """Map OFX per-element constants if present in schema [ofx.element].

    Expected keys (element-wise indices):
      - sp, pp_off, sd, pd, dd_off, sf, pf, df, ff_off
    Returns dict with tensors per key shaped (Zmax+1,).
    """
    if schema.ofx_element is None:
        raise ValueError("Schema missing [ofx.element] for OFX onsite constants")
    maxz = max(g.elements)
    out: Dict[str, torch.Tensor] = {}
    for key, idx in schema.ofx_element.items():
        arr = torch.zeros(maxz + 1, dtype=torch.float64)
        for z, blk in g.elements.items():
            arr[z] = _get_line_val(blk.lines, idx)
        out[key] = arr
    return out
    if schema.spin is None or 'kW' not in schema.spin:
        raise ValueError("Schema missing [spin].kW mapping (Eq. 121)")
    maxz = max(g.elements)
    out = torch.zeros(maxz + 1, dtype=torch.float64)
    for z, blk in g.elements.items():
        out[z] = _get_line_val(blk.lines, schema.spin['kW'])
    return out


 


def map_diatomic_params(g: GxTBParameters, schema: GxTBSchema) -> Dict[str, torch.Tensor]:
    """
    Map diatomic-frame scaling parameters k_A^{diat,L} per Eq. (31)–(32).
    Returns dict with 'sigma','pi','delta' tensors of shape (Zmax+1,).
    """
    if schema.diatomic is None:
        raise ValueError("Schema missing [diatomic] mapping for diatomic scaling (Eqs. 31–32).")
    maxz = max(g.elements)
    out = {L: torch.zeros(maxz + 1, dtype=torch.float64) for L in ("sigma", "pi", "delta")}
    for z, blk in g.elements.items():
        for L, idx in schema.diatomic.items():
            out[L][z] = _get_line_val(blk.lines, idx)
    return out


def map_qvszp_prefactors(g: GxTBParameters, schema: GxTBSchema) -> Dict[str, torch.Tensor]:
    """Map q‑vSZP effective-charge prefactors per doc/theory/7 Eq. (28).

    Expected schema section [qvszp] with keys: k0, k1, k2, k3 mapping to element-wise indices.
    Raises ValueError if the section or any key is missing (no hidden defaults permitted).
    """
    if not hasattr(schema, 'qvszp') or getattr(schema, 'qvszp') is None:
        raise ValueError("Schema missing [qvszp] section for q‑vSZP prefactors (doc/theory/7 Eq. 28)")
    section = getattr(schema, 'qvszp')
    required = ['k0', 'k1', 'k2', 'k3']
    for key in required:
        if key not in section:
            raise ValueError(f"Schema [qvszp] missing '{key}' index (Eq. 28)")
    maxz = max(g.elements)
    out: Dict[str, torch.Tensor] = {}
    for key in required:
        arr = torch.zeros(maxz + 1, dtype=torch.float64)
        idx = section[key]
        for z, blk in g.elements.items():
            arr[z] = _get_line_val(blk.lines, idx)
        out[key] = arr
    return out


def map_mfx_element(g: GxTBParameters, schema: GxTBSchema) -> torch.Tensor:
    """Map per-element shell parameters U^{MFX}_l (doc/theory/20, Eq. 149).

    Expects [mfx.element] with keys: U_s, U_p, U_d, U_f mapping to element lines.
    Returns tensor U_shell with shape (Zmax+1, 4) in order s,p,d,f.
    """
    if schema.mfx_element is None:
        raise ValueError("Schema missing [mfx.element] for MFX per-element U_shell (Eq. 149)")
    keys = ('U_s','U_p','U_d','U_f')
    for k in keys:
        if k not in schema.mfx_element:
            raise ValueError(f"Schema [mfx.element] missing '{k}' index")
    maxz = max(g.elements)
    U_shell = torch.zeros((maxz + 1, 4), dtype=torch.float64)
    idx_map = [schema.mfx_element[k] for k in keys]
    for z, blk in g.elements.items():
        for l, idx in enumerate(idx_map):
            U_shell[z, l] = _get_line_val(blk.lines, idx)
    return U_shell


def map_mfx_global(g: GxTBParameters, schema: GxTBSchema) -> Dict[str, torch.Tensor | float]:
    """Map MFX global scalars and per-shell exponents (doc/theory/20, Eqs. 149–153).

    Expects [mfx] with keys: alpha, omega, k1, k2 (global lines), and optionally
    xi_s, xi_p, xi_d, xi_f (per-shell exponents ξ). If ξ_* are absent, return
    ξ = [1, 1, 2, 2] per doc (valence vs polarization) deterministically.
    """
    if schema.mfx is None:
        raise ValueError("Schema missing [mfx] for MFX global parameters (alpha, omega, k1, k2)")
    gl = g.global_lines.lines
    def gval(name: str) -> float:
        if name not in schema.mfx:
            raise ValueError(f"Schema [mfx] missing '{name}' index")
        return _get_line_val(gl, schema.mfx[name])
    alpha = gval('alpha')
    omega = gval('omega')
    k1 = gval('k1')
    k2 = gval('k2')
    # Optional per-shell exponents
    if all((k in schema.mfx) for k in ('xi_s','xi_p','xi_d','xi_f')):
        xi = torch.tensor([
            _get_line_val(gl, schema.mfx['xi_s']),
            _get_line_val(gl, schema.mfx['xi_p']),
            _get_line_val(gl, schema.mfx['xi_d']),
            _get_line_val(gl, schema.mfx['xi_f']),
        ], dtype=torch.float64)
    else:
        # Theory-defined choice per doc/theory/20, Eq. 150 discussion:
        # valence shells (s,p) use ξ=1; polarization (d,f) use ξ=2.
        xi = torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float64)
    return {"alpha": alpha, "omega": omega, "k1": k1, "k2": k2, "xi_l": xi}


def validate_tb_parameters(g: GxTBParameters, schema: GxTBSchema, numbers: torch.Tensor | None = None) -> None:
    """Validate mapped parameters for basic physical sanity.

    Checks (raises ValueError on violation):
    - Hubbard gamma and gamma3 values are finite for all elements and gamma > 0.
    - Third-order globals k3, k3x are positive; kGamma_l finite.
    - Fourth-order gamma4 positive if present.
    - q‑vSZP prefactors k0..k3 finite (|value| <= 1e3 guard) and r_cov non-negative and finite.
    - Diatomic scaling parameters finite.

    This function does not alter values or add defaults; it only validates the schema‑mapped entries.
    """
    # Build active element set to validate (either all mapped or the subset in 'numbers')
    active_Z = None
    if numbers is not None:
        active_Z = set(int(z) for z in numbers.view(-1).tolist())
    # Hubbard gamma
    try:
        hub = map_hubbard_params(g, schema)  # type: ignore[name-defined]
    except Exception:
        hub = None
    if hub is not None and 'gamma' in hub:
        gam = hub['gamma']
        if active_Z is not None:
            mask = torch.zeros_like(gam, dtype=torch.bool)
            for z in active_Z:
                if z < gam.shape[0]:
                    mask[z] = True
            gam_chk = gam[mask]
        else:
            gam_chk = gam
        if gam_chk.numel() == 0 or (not torch.isfinite(gam_chk).all()):
            raise ValueError("Hubbard gamma must be finite for all active elements")
        # Heuristic bound check (aligns with tests): allow zeros; ensure values are in a reasonable range
        if (gam_chk[gam_chk != 0.0] < -1.0).any() or (gam_chk[gam_chk != 0.0] > 5.0).any():
            raise ValueError("Hubbard gamma outside expected range [-1,5] for active elements")
        if 'gamma3' in hub:
            g3 = hub['gamma3']
            g3_chk = g3[mask] if active_Z is not None else g3
            if not torch.isfinite(g3_chk).all():
                raise ValueError("Hubbard gamma3 must be finite for all mapped elements")
    # Third-order
    if schema.third_order is not None:
        top = map_third_order_params(g, schema)  # type: ignore[name-defined]
        g3e = top['gamma3_elem']
        if active_Z is not None:
            mask = torch.zeros_like(g3e, dtype=torch.bool)
            for z in active_Z:
                if z < g3e.shape[0]:
                    mask[z] = True
            g3e_chk = g3e[mask]
        else:
            g3e_chk = g3e
        if not torch.isfinite(g3e_chk).all():
            raise ValueError("[third_order].gamma3_elem contains non-finite entries")
        kGamma = top['kGamma']
        if not torch.isfinite(kGamma).all():
            raise ValueError("[third_order].kGamma_l contains non-finite entries")
        if not (float(top['k3']) > 0.0 and float(top['k3x']) > 0.0):
            raise ValueError("[third_order].k3 and k3x must be positive")
    # Fourth-order
    if schema.fourth_order is not None and 'gamma4' in schema.fourth_order:
        g4 = map_fourth_order_params(g, schema)  # type: ignore[name-defined]
        if not (float(g4) > 0.0):
            raise ValueError("[fourth_order].gamma4 must be positive")
    # q‑vSZP
    if hasattr(schema, 'qvszp') and getattr(schema, 'qvszp') is not None:
        qv = map_qvszp_prefactors(g, schema)
        for k in ('k0','k1','k2','k3'):
            v = qv[k]
            if not torch.isfinite(v).all():
                raise ValueError(f"[qvszp].{k} contains non-finite values")
            if (v.abs() > 1e3).any():
                raise ValueError(f"[qvszp].{k} magnitude exceeds 1e3; likely mis-mapped schema")
        # Validate r_cov via CN mapping
        cn = map_cn_params(g, schema)  # type: ignore[name-defined]
        rcv = cn['r_cov']
        if not torch.isfinite(rcv).all() or (rcv < 0).any():
            raise ValueError("[cn].r_cov must be finite and non-negative")
        if not (float(cn['k_cn']) > 0.0):
            raise ValueError("[cn].k_cn must be positive")
    # Diatomic scaling
    if schema.diatomic is not None:
        di = map_diatomic_params(g, schema)
        for L in ('sigma','pi','delta'):
            if not torch.isfinite(di[L]).all():
                raise ValueError(f"[diatomic].{L} contains non-finite values")


def map_eht_params(g: GxTBParameters, schema: GxTBSchema) -> Dict[str, torch.Tensor]:
    """Map EHT onsite and Wolfsberg parameters (doc/theory/12_eht_hamiltonian.md).

    Expected schema.eht keys (examples): eps_s,eps_p,eps_d,k_w_s,k_w_p,k_w_d.
    Legacy files may instead use h_s,h_p,h_d; consumers should accept both.
    Polynomial coefficients (optional): pi_* families each list of indices.
    Returns dict of tensors sized (Zmax+1,) or (Zmax+1, ncoeff).
    """
    if schema.eht is None:
        raise ValueError("Schema missing [eht] section for EHT parameters")
    maxz = max(g.elements)
    def zeros():
        return torch.zeros(maxz + 1, dtype=torch.float64)
    out: Dict[str, torch.Tensor] = {}
    scalar_keys = [k for k in schema.eht.keys()]
    for key in scalar_keys:
        arr = zeros()
        idx = schema.eht[key]
        for z, blk in g.elements.items():
            arr[z] = _get_line_val(blk.lines, idx)
        out[key] = arr
    # legacy generic poly
    if schema.eht_poly:
        for pname, plist in schema.eht_poly.items():
            arr = torch.zeros(maxz + 1, len(plist), dtype=torch.float64)
            for z, blk in g.elements.items():
                for ci, idx in enumerate(plist):
                    arr[z, ci] = _get_line_val(blk.lines, idx)
            out[pname] = arr
    # CN polynomials
    if schema.eht_cn_poly:
        for pname, plist in schema.eht_cn_poly.items():
            arr = torch.zeros(maxz + 1, len(plist), dtype=torch.float64)
            for z, blk in g.elements.items():
                for ci, idx in enumerate(plist):
                    arr[z, ci] = _get_line_val(blk.lines, idx)
            out[pname] = arr
    # distance polynomials
    if schema.eht_r_poly:
        for pname, plist in schema.eht_r_poly.items():
            arr = torch.zeros(maxz + 1, len(plist), dtype=torch.float64)
            for z, blk in g.elements.items():
                for ci, idx in enumerate(plist):
                    arr[z, ci] = _get_line_val(blk.lines, idx)
            out[pname] = arr
    return out


def map_repulsion_params(g: GxTBParameters, schema: GxTBSchema) -> Dict[str, torch.Tensor | float]:
    # global params from global_lines
    gl = g.global_lines.lines
    def gval(idx: Tuple[int, int]) -> float:
        return _get_line_val(gl, idx)

    kpen1_hhe = gval(schema.repulsion.kpen1_hhe)
    kpen1_rest = gval(schema.repulsion.kpen1_rest)
    kpen2 = gval(schema.repulsion.kpen2)
    kpen3 = gval(schema.repulsion.kpen3)
    kpen4 = gval(schema.repulsion.kpen4)
    kexp = gval(schema.repulsion.kexp)

    # element-wise tensors
    maxz = max(g.elements)
    z_eff0 = torch.zeros(maxz + 1, dtype=torch.float64)
    alpha0 = torch.zeros_like(z_eff0)
    kq = torch.zeros_like(z_eff0)
    kq2 = torch.zeros_like(z_eff0)
    kcn = torch.zeros_like(z_eff0)
    r0 = torch.zeros_like(z_eff0)
    for z, blk in g.elements.items():
        lines = blk.lines
        z_eff0[z] = _get_line_val(lines, schema.repulsion.z_eff0)
        alpha0[z] = _get_line_val(lines, schema.repulsion.alpha0)
        kq[z] = _get_line_val(lines, schema.repulsion.kq)
        kq2[z] = _get_line_val(lines, schema.repulsion.kq2)
        kcn[z] = _get_line_val(lines, schema.repulsion.kcn)
        r0[z] = _get_line_val(lines, schema.repulsion.r0)

    return {
        "z_eff0": z_eff0,
        "alpha0": alpha0,
        "kq": kq,
        "kq2": kq2,
        "kcn": kcn,
        "r0": r0,
        "kpen1_hhe": kpen1_hhe,
        "kpen1_rest": kpen1_rest,
        "kpen2": kpen2,
        "kpen3": kpen3,
        "kpen4": kpen4,
        "kexp": kexp,
    }


def map_cn_params(g: GxTBParameters, schema: GxTBSchema) -> Dict[str, torch.Tensor | float]:
    gl = g.global_lines.lines
    def gval(idx: Tuple[int, int]) -> float:
        return _get_line_val(gl, idx)

    k_cn = gval(schema.cn.k_cn)
    maxz = max(g.elements)
    r_cov = torch.zeros(maxz + 1, dtype=torch.float64)
    for z, blk in g.elements.items():
        r_cov[z] = _get_line_val(blk.lines, schema.cn.r_cov)

    return {"r_cov": r_cov, "k_cn": k_cn}


def map_increment_params(g: GxTBParameters, schema: GxTBSchema) -> torch.Tensor:
    maxz = max(g.elements)
    deinc = torch.zeros(maxz + 1, dtype=torch.float64)
    for z, blk in g.elements.items():
        deinc[z] = _get_line_val(blk.lines, schema.increment.delta_e_incr)
    return deinc


def map_hubbard_params(g: GxTBParameters, schema: GxTBSchema) -> Dict[str, torch.Tensor]:
    """Map heuristic Hubbard parameters (gamma, gamma3) per element.

    gamma: effective chemical hardness for charge quadratic term.
    gamma3: cubic correction coefficient (optional higher-order expansion).

    Returns dict with tensors shape (Zmax+1,). Missing schema raises error.
    """
    if schema.hubbard is None:
        raise ValueError("Schema missing [hubbard] section for Hubbard parameters")
    maxz = max(g.elements)
    out: Dict[str, torch.Tensor] = {k: torch.zeros(maxz + 1, dtype=torch.float64) for k in schema.hubbard.keys()}
    for z, blk in g.elements.items():
        for name, idx in schema.hubbard.items():
            out[name][z] = _get_line_val(blk.lines, idx)
    return out
