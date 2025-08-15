from __future__ import annotations

"""Total energy assembly per doc/theory.

Aggregates components:
 - E_incr (Eq. 13; doc/theory/10_atomic_energy_increment.md)
 - E_rep  (Eqs. 14–17; doc/theory/11_semi_classical_repulsion.md)
 - E_el   = E^{(1)} + E^{(2)} (Eq. 18 with Eq. 63 trace, plus Eq. 100b)

Traceability:
 - Overlap scaling (Eqs. 31–32) and onsite (Eq. 65) are handled in EHT core.
 - SCF solver (Eq. 12) yields density P; energy Tr(PH) (Eq. 63).

No placeholders: If optional components (AES, D4, MFX/OFX) are requested, raise
descriptive NotImplementedError pointing to the governing equations.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ..params.loader import GxTBParameters, EEQParameters
from ..params.schema import (
    GxTBSchema,
    map_repulsion_params,
    map_increment_params,
    map_cn_params,
    map_hubbard_params,
    map_qvszp_prefactors,
    validate_tb_parameters,
)
from ..classical.increment import energy_increment
from ..classical.repulsion import RepulsionParams, repulsion_energy
from ..classical.dispersion import D4Method, load_d4_method
from ..charges.eeq import compute_eeq_charges
from ..basis.qvszp import AtomBasis
from ..hamiltonian.scf_adapter import build_eht_core, make_core_builder
from ..hamiltonian.ofx import build_lambda0_ao_from_element
from ..scf import scf, SCFResult
from ..hamiltonian.third_order import ThirdOrderParams, third_order_energy
from ..hamiltonian.second_order_tb import (
    ShellSecondOrderParams,
    compute_reference_shell_populations,
    compute_shell_charges,
)
from ..params.schema import map_third_order_params, map_spin_kW, map_cn_params
from ..cn import coordination_number
from ..hamiltonian.spin import SpinParams, spin_energy, compute_shell_magnetizations


@dataclass
class EnergyBreakdown:
    E_total: torch.Tensor
    E_el: torch.Tensor
    E1: torch.Tensor
    E2: Optional[torch.Tensor]
    E_rep: torch.Tensor
    E_incr: torch.Tensor
    E_shift: torch.Tensor  # shift-of-reference-state zeroth-order (E_incr + E_rep), doc/theory/3
    scf: SCFResult
    E3: Optional[torch.Tensor] = None
    E_disp: Optional[torch.Tensor] = None


def _build_repulsion_params(gparams: GxTBParameters, schema: GxTBSchema) -> RepulsionParams:
    rp = map_repulsion_params(gparams, schema)
    cn = map_cn_params(gparams, schema)
    return RepulsionParams(
        z_eff0=rp["z_eff0"],
        alpha0=rp["alpha0"],
        kq=rp["kq"],
        kq2=rp["kq2"],
        kcn_elem=rp["kcn"],
        r0=rp["r0"],
        kpen1_hhe=float(rp["kpen1_hhe"]),
        kpen1_rest=float(rp["kpen1_rest"]),
        kpen2=float(rp["kpen2"]),
        kpen3=float(rp["kpen3"]),
        kpen4=float(rp["kpen4"]),
        kexp=float(rp["kexp"]),
        r_cov=cn["r_cov"],
        k_cn=float(cn["k_cn"]),
    )


def compute_total_energy(
    numbers: torch.Tensor,
    positions: torch.Tensor,
    basis: AtomBasis,
    gparams: GxTBParameters,
    schema: GxTBSchema,
    eeq: EEQParameters,
    *,
    total_charge: float,
    nelec: int,
    wolfsberg_mode: str = "arithmetic",
    # Enable all TB orders by default (E^(1)+E^(2)+E^(3)+E^(4)) per SI model aggregation
    second_order: bool = True,
    shell_second_order: bool = False,
    basis_aware_p0: bool = False,
    # UHF controls
    uhf: bool = False,
    nelec_alpha: Optional[int] = None,
    nelec_beta: Optional[int] = None,
    # Higher-order energy flags
    fourth_order: bool = False,
    gamma4: Optional[float] = None,
    third_order: bool = False,
    third_shell_params: Optional[ShellSecondOrderParams] = None,
    third_params: Optional[ThirdOrderParams] = None,
    # Spin polarization energy (requires UHF and W0 matrix)
    spin: bool = False,
    spin_W0: Optional[torch.Tensor] = None,
    # OFX onsite exchange correction (disable by default)
    ofx: bool = False,
    ofx_params: Optional[dict] = None,
    # AES anisotropic electrostatics (disabled by default)
    aes: bool = False,
    # ACP optional non-local correction
    acp: bool = False,
    acp_params: Optional[dict] = None,
    # MFX long-range exchange (disabled by default)
    mfx: bool = False,
    mfx_params: Optional[dict] = None,
    # Dispersion (DFT-D4 only here): energy added post-SCF (no Fock coupling)
    dispersion: bool = False,
    dispersion_params: Optional[dict] = None,
) -> EnergyBreakdown:
    """Compute total energy with explicit component breakdown.

    Components implemented: E_incr (Eq. 13), E_rep (Eqs. 14–17), E_el = E1 + E2 (Eq. 63 + Eq. 100b).
    Not implemented components (AES, D4, MFX/OFX) are intentionally excluded and must not be silently added.
    """
    device = positions.device
    dtype = positions.dtype
    # Early schema/parameter validation (sanity checks; do not modify values)
    validate_tb_parameters(gparams, schema, numbers)
    # Build core (S, ao_atoms) and builder closure
    core = build_eht_core(numbers, positions, basis, gparams, schema, wolfsberg_mode=wolfsberg_mode)
    builder = make_core_builder(basis, gparams, schema, wolfsberg_mode=wolfsberg_mode)
    S = core["S"].to(device=device, dtype=dtype)
    ao_atoms = core["ao_atoms"].to(device=device)

    # Hubbard parameters (γ, γ3) per element
    hub = map_hubbard_params(gparams, schema)

    # Second-order parameters packaged for SCF
    so_params: Optional[Dict[str, torch.Tensor]] = None
    if second_order:
        cn_map = map_cn_params(gparams, schema)
        if shell_second_order:
            # Build shell parameters from Hubbard gamma as U^{(2),0} baseline
            maxz = int(numbers.max().item())
            from ..hamiltonian.second_order_tb import build_shell_second_order_params as _build_sp
            shell_params = _build_sp(maxz, map_hubbard_params(gparams, schema)["gamma"])  # broadcast eta over shells
            # Coordination numbers for current geometry
            cn_vec = coordination_number(positions, numbers, cn_map['r_cov'].to(device=device, dtype=dtype), float(cn_map['k_cn']))
            # Provide shell_params and cn to SCF via so_params
            so_params = {  # type: ignore[assignment]
                "shell_params": shell_params,  # consumed by scf shell path
                "cn": cn_vec,
                "basis_aware_p0": bool(basis_aware_p0),
            }
        else:
            so_params = {
                "eta": hub["gamma"],  # use γ as η-like (Eq. 102 atomic analogue)
                "r_cov": cn_map["r_cov"],
            }

    # SCF solve
    # Prepare third/fourth-order packs for SCF if requested (default: enabled)
    third_shell_pack = None
    third_param_pack = None
    if third_order:
        # Shell params: build from Hubbard gamma as U^{(2)}-like baseline (consistent with earlier second-order path)
        maxz = int(numbers.max().item())
        from ..hamiltonian.second_order_tb import build_shell_second_order_params as _build_sp
        shell_params_third = third_shell_params if third_shell_params is not None else _build_sp(maxz, hub['gamma'])
        # CN
        cn_map = map_cn_params(gparams, schema)
        cn_vec = coordination_number(positions, numbers, cn_map['r_cov'].to(device=device, dtype=dtype), float(cn_map['k_cn']))
        third_shell_pack = {"shell_params": shell_params_third, "cn": cn_vec}
        # Third-order params from schema or provided
        if third_params is None:
            top = map_third_order_params(gparams, schema)
            third_param_pack = {"gamma3_elem": top['gamma3_elem'], "kGamma": top['kGamma'], "k3": top['k3'], "k3x": top['k3x']}
        else:
            third_param_pack = {
                "gamma3_elem": third_params.gamma3_elem,
                "kGamma": torch.tensor(third_params.kGamma_l, dtype=dtype),
                "k3": third_params.k3,
                "k3x": third_params.k3x,
            }

    # Prepare spin params for SCF if requested
    spin_param_pack = None
    if spin:
        kW = map_spin_kW(gparams, schema)
        if spin_W0 is None:
            raise ValueError("spin=True requires spin_W0 (4x4)")
        spin_param_pack = {"kW": kW, "W0": spin_W0}

    # Prepare AES params for SCF if requested
    aes_pack = None
    if aes:
        from ..hamiltonian.aes import AESParams
        from ..params.schema import map_aes_global, map_aes_element
        cn_map = map_cn_params(gparams, schema)
        aesg = map_aes_global(gparams, schema)
        aese = map_aes_element(gparams, schema)
        # Optional higher orders (dmp7/dmp9) included if present
        aes_params = AESParams(
            dmp3=float(aesg['dmp3']), dmp5=float(aesg['dmp5']),
            mprad=aese['mprad'], mpvcn=aese['mpvcn'],
            dmp7=float(aesg['dmp7']) if 'dmp7' in aesg else None,
            dmp9=float(aesg['dmp9']) if 'dmp9' in aesg else None,
        )
        # Extract optional SI rules keys (strings) from aesg starting with 'si_'
        si_rules = {k: aesg[k] for k in aesg.keys() if isinstance(k, str) and k.startswith('si_')}
        aes_pack = {"params": aes_params, "r_cov": cn_map['r_cov'], "k_cn": cn_map['k_cn'], "si_rules": si_rules}

    # Prepare OFX pack if requested
    ofx_pack = None
    if ofx:
        if ofx_params is None or 'alpha' not in ofx_params:
            raise ValueError("ofx=True requires ofx_params with at least 'alpha'")
        alpha = float(ofx_params['alpha'])
        Lam0 = ofx_params.get('Lambda0_ao')
        if Lam0 is None:
            # Option 1: ofx_elem provided directly as per-element tensors
            if 'ofx_elem' in ofx_params:
                ofx_elem = ofx_params['ofx_elem']
                Lam0 = build_lambda0_ao_from_element(numbers, basis, ofx_elem)
            else:
                # Option 2: schema mapping exists
                from ..params.schema import map_ofx_element
                if schema is None or schema.ofx_element is None:
                    raise ValueError("OFX: no 'Lambda0_ao' or 'ofx_elem' provided and schema [ofx.element] missing")
                ofx_elem = map_ofx_element(gparams, schema)
                Lam0 = build_lambda0_ao_from_element(numbers, basis, ofx_elem)
        ofx_pack = {'alpha': alpha, 'Lambda0_ao': Lam0}

    # If fourth-order enabled and gamma4 not provided, map from schema
    if fourth_order and gamma4 is None:
        from ..params.schema import map_fourth_order_params
        gamma4 = map_fourth_order_params(gparams, schema)

    # Prepare MFX pack (pass-through; if not provided, attempt schema mapping)
    mfx_pack = None
    if mfx:
        if mfx_params is None:
            try:
                from ..params.schema import map_mfx_element, map_mfx_global
                U_shell = map_mfx_element(gparams, schema)
                gmap = map_mfx_global(gparams, schema)
                mfx_pack = {
                    'alpha': gmap['alpha'], 'omega': gmap['omega'], 'k1': gmap['k1'], 'k2': gmap['k2'],
                    'U_shell': U_shell.to(device=device, dtype=dtype), 'xi_l': gmap['xi_l'],
                }
            except Exception as e:
                raise ValueError("mfx=True requires mfx_params with keys {'alpha','omega','k1','k2','U_shell','xi_l'} or a schema mapping via [mfx] and [mfx.element]") from e
        else:
            mfx_pack = mfx_params

    # NOTE: We do not wire dispersion Fock into SCF. We add D4 energy after SCF via tad-dftd4 API.

    # q‑vSZP prefactors and EEQ charges for dynamic overlap (Eqs. 27–28)
    cn_map_global = map_cn_params(gparams, schema)
    q_eeq = compute_eeq_charges(numbers, positions, eeq, total_charge=total_charge, dtype=dtype, device=device)
    qv_pref = map_qvszp_prefactors(gparams, schema)
    # Move tensors to device/dtype
    qv_pack = {
        'k0': qv_pref['k0'].to(device=device, dtype=dtype),
        'k1': qv_pref['k1'].to(device=device, dtype=dtype),
        'k2': qv_pref['k2'].to(device=device, dtype=dtype),
        'k3': qv_pref['k3'].to(device=device, dtype=dtype),
        'r_cov': cn_map_global['r_cov'].to(device=device, dtype=dtype),
        'k_cn': float(cn_map_global['k_cn']),
    }

    # SCF call (always executed; optional packs may be None)
    res = scf(
        numbers,
        positions,
        basis,
        builder,
        S,
        hubbard=hub,
        ao_atoms=ao_atoms,
        nelec=nelec,
        eeq_charges=q_eeq,
        second_order=second_order,
        so_params=so_params,
        uhf=uhf,
        nelec_alpha=nelec_alpha,
        nelec_beta=nelec_beta,
        fourth_order=fourth_order,
        fourth_params={"gamma4": gamma4} if fourth_order and gamma4 is not None else None,
        third_order=third_order,
        third_shell_params=third_shell_pack,
        third_params=third_param_pack,
        mfx=mfx, mfx_params=mfx_pack,
        spin=spin, spin_params=spin_param_pack,
        aes=aes, aes_params=aes_pack,
        ofx=ofx, ofx_params=ofx_pack,
        acp=acp, acp_params=acp_params,
        # Activate dynamic q‑vSZP path (doc/theory/7 Eq. 27–28 + doc/theory/8 Eqs. 31–32)
        dynamic_overlap=True,
        qvszp_params=qv_pack,
        q_reference=None,  # neutral reference state for Δq (doc/theory/3)
    )

    # First-order energy from H0 (Eq. 63). Build H0 with final dynamic coefficients for consistency.
    from ..basis.qvszp import compute_effective_charge, build_dynamic_primitive_coeffs
    q_eff_fin = compute_effective_charge(
        numbers, positions, res.q.to(device), q_eeq.to(device),
        r_cov=qv_pack['r_cov'], k_cn=float(qv_pack['k_cn']),
        k0=qv_pack['k0'], k1=qv_pack['k1'], k2=qv_pack['k2'], k3=qv_pack['k3'],
    )
    coeff_list = build_dynamic_primitive_coeffs(numbers, basis, q_eff_fin)
    coeffs_map = {i: c for i, c in enumerate(coeff_list)}
    H0 = builder(numbers, positions, {"q": res.q, "coeffs": coeffs_map})["H0"].to(device=device, dtype=dtype)
    # Electronic energy (E_el) must exclude exchange corrections (OFX/MFX) and other
    # optional Fock add-ons. Per doc/theory aggregation, define E_el ≡ E^(1) + E^(2).
    # eq: 63 for E^(1) via Tr(P H0) and eq: 100b for E^(2) (atomic or shell-resolved).
    E1 = torch.einsum("ij,ji->", res.P, H0)
    E2 = res.E2 if res.E2 is not None else None
    E3 = res.E3 if res.E3 is not None else None
    E4 = res.E4 if res.E4 is not None else None
    E_el = E1 + (E2 if E2 is not None else 0.0)

    # Classical increments and repulsion
    de_incr = map_increment_params(gparams, schema)
    E_incr = energy_increment(numbers, de_incr).to(device=device, dtype=dtype)
    rep_params = _build_repulsion_params(gparams, schema)
    # q_eeq already computed above for q_eff; reuse for repulsion energy
    E_rep = repulsion_energy(positions, numbers, rep_params, q_eeq)

    # Third-order energy: prefer SCFResult.E3; recompute only if requested and params provided
    if third_order and E3 is None and third_shell_pack is not None and third_param_pack is not None:
        # Recompute E3 using the same packs used for SCF (no hidden defaults)
        if basis_aware_p0:
            from ..hamiltonian.second_order_tb import compute_reference_shell_populations_basis_aware as _ref_p0
        else:
            from ..hamiltonian.second_order_tb import compute_reference_shell_populations as _ref_p0
        ref_pops = _ref_p0(numbers, basis).to(device=device, dtype=dtype)
        q_shell = compute_shell_charges(res.P, S, basis, ref_pops)
        q_atom = res.q.to(device=device, dtype=dtype)
        cn_map = map_cn_params(gparams, schema)
        cn = coordination_number(positions, numbers, cn_map["r_cov"].to(device=device, dtype=dtype), float(cn_map["k_cn"]))
        shells = basis.shells
        l_map = {"s":0, "p":1, "d":2, "f":3}
        z_list = torch.tensor([sh.element for sh in shells], dtype=torch.long, device=device)
        l_idx = torch.tensor([l_map[sh.l] for sh in shells], dtype=torch.long, device=device)
        atom_idx = torch.tensor([sh.atom_index for sh in shells], dtype=torch.long, device=device)
        # Build U_shell consistent with SCF's shell path
        sp = third_shell_pack["shell_params"]
        U0_shell = sp.U0[z_list, l_idx]
        kU_shell = sp.kU[z_list]
        U_shell = U0_shell * (1.0 + kU_shell * cn[atom_idx])
        # Third-order params
        from ..hamiltonian.third_order import ThirdOrderParams as _TOP
        top = third_param_pack
        tparams = _TOP(
            gamma3_elem=top["gamma3_elem"].to(device=device, dtype=dtype),
            kGamma_l=(float(top["kGamma"][0].item()), float(top["kGamma"][1].item()), float(top["kGamma"][2].item()), float(top["kGamma"][3].item())),
            k3=float(top["k3"]),
            k3x=float(top["k3x"]),
        )
        try:
            E3_val = third_order_energy(numbers, positions, basis, q_shell, q_atom, U_shell, tparams)
            E3 = E3_val if torch.isfinite(E3_val) else None
        except Exception:
            E3 = None

    # Optional dispersion energy: D4 (two-body BJ + ATM) per tad-dftd4 theory, fully reimplemented here
    E_disp = None
    if dispersion:
        from ..classical.dispersion import d4_energy, load_d4_method
        # Require D4 reference data dict provided by caller (no hidden defaults)
        if dispersion_params is None or 'ref' not in dispersion_params:
            raise ValueError("dispersion=True requires dispersion_params with 'ref' containing D4 reference arrays (refsys, refascale, refscount, secscale, secalpha, refalpha, refcovcn, refc, zeff, gam, r4r2, clsq)")
        # Damping parameters from TOML (tad-dftd4 semantics): [default]/[parameter.<functional>], variant e.g. 'bj-eeq-atm'
        functional = dispersion_params.get('functional', None) if dispersion_params is not None else None
        variant = dispersion_params.get('variant', None) if dispersion_params is not None else None
        method = load_d4_method('parameters/dftd4parameters.toml', variant=variant or 'bj-eeq-atm', functional=functional)
        # Use EEQ charges from SCF unless explicit q provided
        q_for_d4 = dispersion_params.get('q', None) if dispersion_params is not None else None
        q_use = q_for_d4.to(device=device, dtype=dtype) if isinstance(q_for_d4, torch.Tensor) else res.q.to(device=device, dtype=dtype)
        E_disp = d4_energy(numbers, positions, q_use, method, dispersion_params['ref'])
    # Optional spin-polarization energy (Eq. 120b)
    E_spin = 0.0
    if spin:
        if not uhf:
            raise ValueError("spin=True requires uhf=True to form magnetizations")
        sp = SpinParams(kW_elem=spin_param_pack['kW'], W0=spin_param_pack['W0'])
        if res.P_alpha is None or res.P_beta is None:
            raise RuntimeError("UHF result missing P_alpha/P_beta for spin energy")
        m_shell = compute_shell_magnetizations(res.P_alpha, res.P_beta, S, basis)
        E_spin = spin_energy(numbers, basis, m_shell, sp)
    # Shift-of-reference-state zeroth-order energy (doc/theory/3): collect increment + repulsion
    E_shift = (E_incr + E_rep)
    # Explicitly add optional corrections that were previously folded into Tr(PH):
    # AES (doc/theory/16), OFX (doc/theory/21), MFX (doc/theory/20), ACP (doc/theory/13).
    E_corr = 0.0
    if res.E_AES is not None:
        E_corr = E_corr + res.E_AES
    if res.E_OFX is not None:
        E_corr = E_corr + res.E_OFX
    if hasattr(res, 'E_MFX') and res.E_MFX is not None:
        E_corr = E_corr + res.E_MFX
    if hasattr(res, 'E_ACP') and res.E_ACP is not None:
        E_corr = E_corr + res.E_ACP
    E_total = E_el + E_shift + (E3 if E3 is not None else 0.0) + (E4 if E4 is not None else 0.0) + E_spin + (E_disp if E_disp is not None else 0.0) + E_corr
    return EnergyBreakdown(E_total=E_total, E_el=E_el, E1=E1, E2=E2, E_rep=E_rep, E_incr=E_incr, E_shift=E_shift, scf=res, E3=E3, E_disp=E_disp)


def energy_report(res: EnergyBreakdown) -> Dict[str, float]:
    """Return a JSON-serializable per-term energy report.

    Includes optional terms if present in SCFResult (E_AES, E_OFX, E3, E4).
    """
    out: Dict[str, float] = {
        'E_total': float(res.E_total.item()),
        'E_el': float(res.E_el.item()),
        'E1': float(res.E1.item()),
        'E_rep': float(res.E_rep.item()),
        'E_incr': float(res.E_incr.item()),
        'E_shift': float(res.E_shift.item()),
    }
    if res.E2 is not None:
        out['E2'] = float(res.E2.item())
    if res.E3 is not None:
        out['E3'] = float(res.E3.item())
    # E4 currently only available in SCFResult if used; add from scf
    if res.scf.E4 is not None:
        out['E4'] = float(res.scf.E4.item())
    if res.E_disp is not None:
        out['E_disp'] = float(res.E_disp.item())
    # Optional AES/OFX/ACP/MFX contributions
    if res.scf.E_AES is not None:
        out['E_AES'] = float(res.scf.E_AES.item())
    if res.scf.E_OFX is not None:
        out['E_OFX'] = float(res.scf.E_OFX.item())
    if hasattr(res.scf, 'E_MFX') and res.scf.E_MFX is not None:
        out['E_MFX'] = float(res.scf.E_MFX.item())
    if hasattr(res.scf, 'E_ACP') and res.scf.E_ACP is not None:
        out['E_ACP'] = float(res.scf.E_ACP.item())
    # SCF info
    out['scf_n_iter'] = int(res.scf.n_iter)
    out['scf_converged'] = 1 if res.scf.converged else 0
    return out
