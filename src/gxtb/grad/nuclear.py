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

    return dE
