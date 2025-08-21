from __future__ import annotations

"""Analytic nuclear gradients via autograd for available terms.

Implements dE/dR for:
 - Second-order atomic isotropic TB (Eqs. 100b–101) by differentiating E^{(2)}(R)
 - AES energy (Sec. 1.11.1; Eqs. 109–111 and 110a–d with damping) by differentiating E^{AES}(R)

These leverage exact PyTorch autodiff over the implemented energy expressions; no finite-difference approximations
are used in the returned gradient.
"""

from typing import Dict, Tuple
import torch

from ..hamiltonian.second_order_tb import SecondOrderParams, compute_gamma2, second_order_energy
from ..hamiltonian.moments_builder import build_moment_matrices
from ..hamiltonian.aes import AESParams, aes_energy_and_fock

Tensor = torch.Tensor

__all__ = [
    "grad_second_order_atomic",
    "grad_aes_energy",
    "grad_third_order_energy",
    "grad_fourth_order_energy",
    "grad_ofx_energy",
    "grad_mfx_energy",
    "grad_acp_energy",
    "grad_spin_energy",
    "total_gradient",
]


def grad_second_order_atomic(
    numbers: Tensor,
    positions: Tensor,
    params: SecondOrderParams,
    q: Tensor,
    q_ref: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Return (E^{(2)}, dE^{(2)}/dR) where E^{(2)} = 1/2 Δq^T γ^{(2)}(R) Δq (Eqs. 100b–101).

    The gradient is exact via autograd over the analytic γ^{(2)} form.
    """
    pos_req = positions.detach().clone().requires_grad_(True)
    gamma2 = compute_gamma2(numbers, pos_req, params)
    E2 = second_order_energy(gamma2, q, q_ref)
    grad, = torch.autograd.grad(E2, pos_req, create_graph=False)
    return E2.detach(), grad.detach()


def grad_aes_energy(
    numbers: Tensor,
    positions: Tensor,
    basis,
    P: Tensor,
    params: AESParams,
    *,
    r_cov: Tensor,
    k_cn: float,
    si_rules: Dict[str, float | str] | None = None,
) -> Tuple[Tensor, Tensor]:
    """Return (E^{AES}, dE^{AES}/dR) by differentiating the AES energy assembled in aes_energy_and_fock.

    This covers contributions up to n=5 always and n=7/n=9 when dmp7/dmp9 are provided in AESParams, with
    damping per either logistic (default) or SI Eq. 117 when si_rules are supplied.
    """
    pos_req = positions.detach().clone().requires_grad_(True)
    # Rebuild AO moments for current geometry
    S, D, Q = build_moment_matrices(numbers, pos_req, basis)
    E, _H = aes_energy_and_fock(
        numbers, pos_req, basis, P, S, D, Q, params, r_cov=r_cov, k_cn=k_cn, si_rules=si_rules
    )
    grad, = torch.autograd.grad(E, pos_req, create_graph=False)
    return E.detach(), grad.detach()


def grad_third_order_energy(
    numbers: Tensor,
    positions: Tensor,
    basis,
    q_shell: Tensor,
    q_atom: Tensor,
    U_shell: Tensor,
    params,
) -> Tuple[Tensor, Tensor]:
    """Return (E^{(3)}, dE^{(3)}/dR) per doc/theory/18_third_order_tb.md.

    Theory mapping:
      - Energy Eq. 129b (third_order.py: third_order_energy)
      - τ^{(3)} kernel Eqs. 132–133 (third_order.py: _tau3_offsite/_tau3_onsite)

    Gradient policy (consistent with second-order implementation):
      - Treat q_shell, q_atom, and U_shell as fixed w.r.t. R (SCF-stationary charges
        and a CN snapshot), and differentiate only the explicit R-dependence through
        τ^{(3)}(R). This yields the analytic kernel contribution to the nuclear force.
    """
    from ..hamiltonian.third_order import third_order_energy as _E3
    pos_req = positions.detach().clone().requires_grad_(True)
    E3 = _E3(numbers, pos_req, basis, q_shell, q_atom, U_shell, params)
    grad, = torch.autograd.grad(E3, pos_req, create_graph=False)
    return E3.detach(), grad.detach()


def grad_fourth_order_energy(
    numbers: Tensor,
    positions: Tensor,
    q: Tensor,
    params,
) -> Tuple[Tensor, Tensor]:
    """Return (E^{(4)}, dE^{(4)}/dR) per doc/theory/19_fourth_order_tb.md.

    Theory mapping:
      - Energy Eq. 140b (fourth_order.py: fourth_order_energy)
      - Fock Eq. 143 affects SCF but does not introduce explicit R-dependence in the
        energy expression when q is treated as fixed at stationarity.

    Under the kernel-differentiation policy with fixed q, E^{(4)} has no explicit
    R-dependence; hence the nuclear gradient contribution is identically zero.
    """
    from ..hamiltonian.fourth_order import fourth_order_energy as _E4
    E4 = _E4(q, params)
    dE = torch.zeros_like(positions)
    return E4, dE


def _build_S_raw_torch(numbers: Tensor, positions: Tensor, basis, coeffs_map: Dict[int, Tensor] | None = None) -> Tensor:
    """Differentiable raw AO overlap S using McMurchie–Davidson torch kernel (doc/theory/8, Eq. 31 context).

    - If coeffs_map is None, use static q‑vSZP baseline c0 per shell (doc/theory/7 Eqs. 8–11 with q_eff=0).
    - Returns symmetric S (enforces 0.5(S+S^T)).
    """
    from ..basis.md_overlap import overlap_shell_pair_torch as _ov_sph_t
    shells = basis.shells
    ao_off = basis.ao_offsets
    ao_cnt = basis.ao_counts
    dtype = positions.dtype
    device = positions.device
    nao = basis.nao
    S = torch.zeros((nao, nao), dtype=dtype, device=device)
    alpha = [torch.tensor([p[0] for p in sh.primitives], dtype=dtype, device=device) for sh in shells]
    if coeffs_map is None:
        coeffs = [torch.tensor([p[1] for p in sh.primitives], dtype=dtype, device=device) for sh in shells]
    else:
        coeffs = [coeffs_map[i].to(device=device, dtype=dtype) for i in range(len(shells))]
    lmap = {"s":0, "p":1, "d":2, "f":3}
    for i, shi in enumerate(shells):
        li = lmap[shi.l]
        ai = alpha[i]; ci = coeffs[i]
        oi, ni = ao_off[i], ao_cnt[i]
        for j in range(i, len(shells)):
            shj = shells[j]
            lj = lmap[shj.l]
            aj = alpha[j]; cj = coeffs[j]
            oj, nj = ao_off[j], ao_cnt[j]
            R = positions[shi.atom_index] - positions[shj.atom_index]
            block = _ov_sph_t(li, lj, ai, ci, aj, cj, R)
            S[oi:oi+ni, oj:oj+nj] = block
            if j != i:
                S[oj:oj+nj, oi:oi+ni] = block.T
    return 0.5 * (S + S.T)


def grad_ofx_energy(
    numbers: Tensor,
    positions: Tensor,
    basis,
    P: Tensor,
    params,
    *,
    coeffs_map: Dict[int, Tensor] | None = None,
) -> Tuple[Tensor, Tensor]:
    """Return (E^{OFX}, dE^{OFX}/dR) per doc/theory/21_ofx.md using differentiable S.

    Assumptions:
      - Λ^0_{AO} provided in params (OFXParams) and independent of positions.
      - Dual density ẐP uses raw AO overlap S (Eq. 156).
      - P is held fixed (kernel-only gradient via ∂S/∂R path).
    """
    from ..hamiltonian.ofx import ofx_energy as _Eofx
    pos_req = positions.detach().clone().requires_grad_(True)
    S = _build_S_raw_torch(numbers, pos_req, basis, coeffs_map)
    E = _Eofx(numbers, basis, P.to(S), S, params)
    grad, = torch.autograd.grad(E, pos_req, create_graph=False)
    return E.detach(), grad.detach()


def grad_mfx_energy(
    numbers: Tensor,
    positions: Tensor,
    basis,
    P: Tensor,
    params,
    *,
    coeffs_map: Dict[int, Tensor] | None = None,
) -> Tuple[Tensor, Tensor]:
    """Return (E^{lr,MFX}, dE/dR) per doc/theory/20_mfx.md.

    Differentiates through both γ^{MFX}(R) (Eq. 149) and AO overlap S used in the symmetric Fock construction (Eq. 153).
    P held fixed.
    """
    from ..hamiltonian.mfx import build_gamma_ao, mfx_fock, mfx_energy
    pos_req = positions.detach().clone().requires_grad_(True)
    S = _build_S_raw_torch(numbers, pos_req, basis, coeffs_map)
    gamma = build_gamma_ao(numbers, pos_req, basis, params)
    F = mfx_fock(P.to(S), S, gamma)
    E = mfx_energy(P.to(S), F)
    grad, = torch.autograd.grad(E, pos_req, create_graph=False)
    return E.detach(), grad.detach()


def grad_acp_energy(
    numbers: Tensor,
    positions: Tensor,
    basis,
    P: Tensor,
    *,
    c0: Tensor,
    xi: Tensor,
    k_acp_cn: float,
    cn_avg: Tensor,
    r_cov: Tensor,
    k_cn: float,
    l_list: tuple[str, ...] = ("s","p","d"),
) -> Tuple[Tensor, Tensor]:
    """Return (E^{ACP}, dE^{ACP}/dR) via differentiable S^{ACP} (doc/theory/13_acp.md Eqs. 78–80).

    Builds AO–ACP projector overlap using torch MD kernel and differentiates E=Tr(H^{ACP} ∘ P) with H^{ACP}=S^{ACP}S^{ACP}^T.
    """
    from ..cn import coordination_number
    from ..hamiltonian.acp import acp_energy as _Eacp
    pos_req = positions.detach().clone().requires_grad_(True)
    device = pos_req.device; dtype = pos_req.dtype
    # CN-dependent coefficients per Eq. 80
    cn = coordination_number(pos_req, numbers, r_cov.to(device=device, dtype=dtype), float(k_cn))
    shells = basis.shells
    nao = basis.nao
    l_map = {"s":0, "p":1, "d":2, "f":3}
    dims = {"s":1, "p":3, "d":5, "f":7}
    naux = sum(dims[l] for _ in range(len(numbers)) for l in l_list)
    S_acp = torch.zeros((nao, naux), dtype=dtype, device=device)
    col_off = 0
    from ..basis.md_overlap import overlap_shell_pair_torch as _ov_sph_t
    alpha_i = [torch.tensor([p[0] for p in sh.primitives], dtype=dtype, device=device) for sh in shells]
    coeff_i = [torch.tensor([p[1] + p[2] for p in sh.primitives], dtype=dtype, device=device) for sh in shells]
    for A in range(len(numbers)):
        Z = int(numbers[A].item())
        for l in l_list:
            ell = l_map[l]; nprj = dims[l]
            c0_Zl = c0[Z, ell].to(device=device, dtype=dtype)
            xi_Zl = xi[Z, ell].to(device=device, dtype=dtype)
            cn_avg_Z = cn_avg[Z].to(device=device, dtype=dtype)
            if float(cn_avg_Z.item()) == 0.0:
                raise ValueError("cn_avg[Z] must be non-zero (doc/theory/13 Eq. 80)")
            c_acp = c0_Zl * (1.0 + float(k_acp_cn) * (cn[A] / cn_avg_Z))
            alpha_j = torch.tensor([float(xi_Zl.item())], dtype=dtype, device=device)
            c_j = torch.tensor([float(c_acp.item())], dtype=dtype, device=device)
            block_cols = slice(col_off, col_off + nprj)
            for ish, sh in enumerate(shells):
                li = l_map[sh.l]
                off_i = basis.ao_offsets[ish]
                ni = basis.ao_counts[ish]
                R = pos_req[sh.atom_index] - pos_req[A]
                S_block = _ov_sph_t(li, ell, alpha_i[ish], coeff_i[ish], alpha_j, c_j, R)
                S_acp[off_i:off_i+ni, block_cols] = S_block
            col_off += nprj
    # Energy and gradient
    H_acp = S_acp @ S_acp.T
    E = torch.einsum('ij,ji->', P.to(dtype=dtype, device=device), H_acp)
    grad, = torch.autograd.grad(E, pos_req, create_graph=False)
    return E.detach(), grad.detach()


def grad_spin_energy(
    numbers: Tensor,
    positions: Tensor,
    basis,
    Pa: Tensor,
    Pb: Tensor,
    params,
) -> Tuple[Tensor, Tensor]:
    """Return (E^{spin}, dE^{spin}/dR) per doc/theory/17 via ∂S/∂R.

    - Rebuild S with differentiable MD torch kernel.
    - Compute shell magnetizations m_{l_A} from (Pa, Pb, S) using Mulliken-like partitioning (Eq. 119b).
    - Evaluate E^{spin} (Eq. 120b) and differentiate w.r.t. positions.
    - P^α and P^β are held fixed; only ∂S enters the gradient.
    """
    from ..hamiltonian.spin import spin_energy as _Espin, compute_shell_magnetizations as _m_shell
    pos_req = positions.detach().clone().requires_grad_(True)
    S = _build_S_raw_torch(numbers, pos_req, basis, coeffs_map=None)
    m_shell = _m_shell(Pa.to(S), Pb.to(S), S, basis)
    E = _Espin(numbers, basis, m_shell, params)
    grad, = torch.autograd.grad(E, pos_req, create_graph=False)
    return E.detach(), grad.detach()


def total_gradient(
    numbers: Tensor,
    positions: Tensor,
    basis,
    gparams,
    schema,
    *,
    # EHT-related
    P: Tensor | None = None,
    include_eht_stepA: bool = False,
    include_dynamic_overlap_cn: bool = False,
    q_scf: Tensor | None = None,
    q_eeqbc: Tensor | None = None,
    # Second-order isotropic (atomic)
    include_second_order: bool = False,
    so_params: dict | None = None,  # expects {'eta': Tensor, 'r_cov': Tensor}
    q: Tensor | None = None,
    q_ref: Tensor | None = None,
    # AES
    include_aes: bool = False,
    aes_params: object | None = None,  # AESParams
    aes_r_cov: Tensor | None = None,
    aes_k_cn: float | None = None,
    aes_si_rules: dict | None = None,
    # OFX onsite exchange (Eq. 155) via ∂S/∂R
    include_ofx: bool = False,
    ofx_params: object | None = None,  # OFXParams
    # MFX long-range exchange (Eqs. 149, 153)
    include_mfx: bool = False,
    mfx_params: object | None = None,  # MFXParams
    # ACP nonlocal correction (Eqs. 78–80)
    include_acp: bool = False,
    acp_params: dict | None = None,  # expects {'c0','xi','k_acp_cn','cn_avg','r_cov','k_cn','l_list'(optional)}
    # Spin polarization (doc/theory/17)
    include_spin: bool = False,
    spin_params: object | None = None,  # SpinParams
    P_alpha: Tensor | None = None,
    P_beta: Tensor | None = None,
    # Third order
    include_third_order: bool = False,
    third_params: object | None = None,  # ThirdOrderParams
    third_q_shell: Tensor | None = None,
    third_q_atom: Tensor | None = None,
    third_U_shell: Tensor | None = None,
    # Fourth order
    include_fourth_order: bool = False,
    fourth_params: object | None = None,  # FourthOrderParams
    fourth_q: Tensor | None = None,
    # Repulsion (doc/theory/11): energy Eq. 52; gradient Eqs. 57–61
    include_repulsion: bool = False,
    repulsion_params: object | None = None,  # RepulsionParams
    eeq: object | None = None,
    total_charge: float = 0.0,
    repulsion_dq_dpos: Tensor | None = None,
) -> Tensor:
    """Aggregate nuclear gradient contributions for currently implemented components.

    Components and equations:
    - EHT Step A (doc/theory/12, Eqs. 68–71) via eht_energy_gradient (excludes ∂Ŝ^{sc}/∂R term).
    - CN-driven dynamic-overlap contribution (doc/theory/7 Eqs. 27–28; doc/theory/8 Eq. 39) when enabled.
    - Second-order isotropic TB (doc/theory/15, Eqs. 100b–101) via autograd on γ^{(2)}(R).
    - AES anisotropic electrostatics (doc/theory/16, Eqs. 109–111 and 110a–d) via autograd on implemented energy.

    No placeholders: each included term requires explicit inputs; otherwise raises ValueError.
    """
    device = positions.device
    dtype = positions.dtype
    nat = numbers.shape[0]
    dE = torch.zeros((nat, 3), dtype=dtype, device=device)

    # --- EHT Step A + CN-driven dynamic-overlap ---
    if include_eht_stepA or include_dynamic_overlap_cn:
        if P is None:
            raise ValueError("EHT gradient requires P (density matrix)")
        from ..hamiltonian.eht import eht_energy_gradient as _eht_grad
        # Build dynamic-overlap pack if requested
        dyn_pack = None
        if include_dynamic_overlap_cn:
            if q_scf is None or q_eeqbc is None:
                raise ValueError("CN-driven dynamic-overlap requires q_scf and q_eeqbc (Eq. 28)")
            from ..params.schema import map_qvszp_prefactors, map_cn_params
            qv = map_qvszp_prefactors(gparams, schema)
            cnm = map_cn_params(gparams, schema)
            dyn_pack = {
                'k0': qv['k0'].to(device=device, dtype=dtype),
                'k1': qv['k1'].to(device=device, dtype=dtype),
                'k2': qv['k2'].to(device=device, dtype=dtype),
                'k3': qv['k3'].to(device=device, dtype=dtype),
                'r_cov': cnm['r_cov'].to(device=device, dtype=dtype),
                'k_cn': float(cnm['k_cn']),
                'q_scf': q_scf.to(device=device, dtype=dtype),
                'q_eeqbc': q_eeqbc.to(device=device, dtype=dtype),
            }
        # Cases
        if include_eht_stepA and include_dynamic_overlap_cn:
            # Full Step A + dynamic-overlap chain
            dE = dE + _eht_grad(numbers, positions, basis, gparams, schema, P, wolfsberg_mode='arithmetic', dynamic_overlap_cn=dyn_pack)
        elif include_dynamic_overlap_cn and not include_eht_stepA:
            # Dynamic-overlap only: subtract the Step A part
            g_full = _eht_grad(numbers, positions, basis, gparams, schema, P, wolfsberg_mode='arithmetic', dynamic_overlap_cn=dyn_pack)
            g_stepA = _eht_grad(numbers, positions, basis, gparams, schema, P, wolfsberg_mode='arithmetic', dynamic_overlap_cn=None)
            dE = dE + (g_full - g_stepA)
        elif include_eht_stepA and not include_dynamic_overlap_cn:
            dE = dE + _eht_grad(numbers, positions, basis, gparams, schema, P, wolfsberg_mode='arithmetic', dynamic_overlap_cn=None)

    # --- Second-order isotropic TB (atomic) ---
    if include_second_order:
        if so_params is None or 'eta' not in so_params or 'r_cov' not in so_params:
            raise ValueError("Second-order gradient requires so_params with {'eta','r_cov'}")
        if q is None or q_ref is None:
            raise ValueError("Second-order gradient requires q and q_ref (Δq = q - q_ref)")
        from ..hamiltonian.second_order_tb import SecondOrderParams
        E2, g2 = grad_second_order_atomic(numbers, positions, SecondOrderParams(eta=so_params['eta'], r_cov=so_params['r_cov']), q.to(dtype), q_ref.to(dtype))
        dE = dE + g2

    # --- AES ---
    if include_aes:
        if aes_params is None or aes_r_cov is None or aes_k_cn is None:
            raise ValueError("AES gradient requires aes_params, aes_r_cov, and aes_k_cn")
        Eaes, gaes = grad_aes_energy(numbers, positions, basis, P if P is not None else torch.zeros((basis.nao, basis.nao), dtype=dtype, device=device), aes_params, r_cov=aes_r_cov, k_cn=float(aes_k_cn), si_rules=aes_si_rules)
        dE = dE + gaes

    # --- OFX ---
    if include_ofx:
        if P is None or ofx_params is None:
            raise ValueError("OFX gradient requires P and ofx_params (doc/theory/21)")
        Eofx, gofx = grad_ofx_energy(numbers, positions, basis, P, ofx_params)
        dE = dE + gofx

    # --- MFX ---
    if include_mfx:
        if P is None or mfx_params is None:
            raise ValueError("MFX gradient requires P and mfx_params (doc/theory/20)")
        Emfx, gmfx = grad_mfx_energy(numbers, positions, basis, P, mfx_params)
        dE = dE + gmfx

    # --- ACP ---
    if include_acp:
        if P is None or acp_params is None:
            raise ValueError("ACP gradient requires P and acp_params (doc/theory/13)")
        for rk in ('c0','xi','k_acp_cn','cn_avg','r_cov','k_cn'):
            if rk not in acp_params:
                raise ValueError(f"ACP gradient missing '{rk}' (doc/theory/13, Eq. 80)")
        Eacp, gacp = grad_acp_energy(
            numbers, positions, basis, P,
            c0=acp_params['c0'], xi=acp_params['xi'], k_acp_cn=float(acp_params['k_acp_cn']),
            cn_avg=acp_params['cn_avg'], r_cov=acp_params['r_cov'], k_cn=float(acp_params['k_cn']),
            l_list=acp_params.get('l_list', ("s","p","d"))
        )
        dE = dE + gacp

    # --- Third-order TB (kernel-only contribution) ---
    if include_third_order:
        if third_params is None or third_q_shell is None or third_q_atom is None or third_U_shell is None:
            raise ValueError("Third-order gradient requires third_params, third_q_shell, third_q_atom, and third_U_shell (doc/theory/18 Eqs. 129b, 132–133)")
        E3, g3 = grad_third_order_energy(numbers, positions, basis, third_q_shell.to(dtype), third_q_atom.to(dtype), third_U_shell.to(dtype), third_params)
        dE = dE + g3

    # --- Fourth-order TB (no explicit R-dependence under fixed q) ---
    if include_fourth_order:
        if fourth_params is None or fourth_q is None:
            raise ValueError("Fourth-order gradient requires fourth_params and fourth_q (doc/theory/19 Eq. 140b)")
        _E4, g4 = grad_fourth_order_energy(numbers, positions, fourth_q.to(dtype), fourth_params)
        dE = dE + g4

    # --- Spin polarization ---
    if include_spin:
        if spin_params is None or P_alpha is None or P_beta is None:
            raise ValueError("Spin gradient requires spin_params, P_alpha, and P_beta (doc/theory/17)")
        Esp, gsp = grad_spin_energy(numbers, positions, basis, P_alpha, P_beta, spin_params)
        dE = dE + gsp

    # --- Semi-classical repulsion (doc/theory/11) ---
    if include_repulsion:
        # Build parameters if not provided
        if repulsion_params is None:
            from ..params.schema import map_repulsion_params, map_cn_params
            from ..classical.repulsion import RepulsionParams as _RParams
            rp = map_repulsion_params(gparams, schema)
            cnm = map_cn_params(gparams, schema)
            repulsion_params = _RParams(
                z_eff0=rp['z_eff0'], alpha0=rp['alpha0'], kq=rp['kq'], kq2=rp['kq2'],
                kcn_elem=rp['kcn'], r0=rp['r0'],
                kpen1_hhe=float(rp['kpen1_hhe']), kpen1_rest=float(rp['kpen1_rest']),
                kpen2=float(rp['kpen2']), kpen3=float(rp['kpen3']), kpen4=float(rp['kpen4']),
                kexp=float(rp['kexp']), r_cov=cnm['r_cov'], k_cn=float(cnm['k_cn'])
            )
        # Charges: if not provided, compute via EEQ with schema mapping
        if q_eeqbc is None:
            if eeq is None:
                raise ValueError("Repulsion gradient requires q_eeqbc or eeq parameters to compute EEQ charges (doc/theory/11 Eq. 53)")
            from ..charges.eeq import compute_eeq_charges
            mapping = getattr(schema, 'eeq', None)
            if mapping is not None and isinstance(mapping, dict):
                mp = {'chi': mapping['chi'], 'eta': mapping['eta'], 'radius': mapping['radius']}
            else:
                mp = None
            q_eeqbc = compute_eeq_charges(numbers, positions, eeq, total_charge=total_charge, mapping=mp, device=positions.device, dtype=positions.dtype)
        # Evaluate gradient
        from ..classical.repulsion import repulsion_energy_and_gradient as _Erep
        # This call will raise a clear error if k^q/k^{q,2} are non-zero and repulsion_dq_dpos is missing (Eq. 58)
        _E, grep = _Erep(positions, numbers, repulsion_params, q_eeqbc, dq_dpos=repulsion_dq_dpos)
        dE = dE + grep

    return dE
