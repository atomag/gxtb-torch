from __future__ import annotations

"""Basic SCF driver (Löwdin + Mulliken) with Hubbard charge shifts for g-xTB.

Features:
 - Optional EEQ baseline charges (passed as eeq_charges).
 - Löwdin orthogonalization X = S^{-1/2}.
 - Mulliken population for charge updates.
 - Linear mixing and Hubbard (gamma, gamma3) onsite corrections.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import torch

Tensor = torch.Tensor

__all__ = [
    "SCFResult",
    "lowdin_orthogonalization",
    "mulliken_charges",
    "scf",
]


@dataclass
class SCFResult:
    H: Tensor
    P: Tensor
    q: Tensor
    q_ref: Tensor
    eps: Tensor
    C: Tensor
    n_iter: int
    converged: bool
    E_elec: Tensor | None = None
    E2: Tensor | None = None  # second-order energy (Eq. 100b) if enabled
    E4: Tensor | None = None  # fourth-order energy (Eq. 140b) if enabled
    E3: Tensor | None = None  # third-order energy (Eq. 129b) if enabled
    E_history: list[float] | None = None
    dq_shell: Tensor | None = None  # shell Δq (Eq. 100a variables) if shell second-order enabled
    V_shell: Tensor | None = None   # shell potentials V^{(2)} (Eq. 106) if available
    P_alpha: Tensor | None = None
    P_beta: Tensor | None = None
    E_AES: Tensor | None = None  # anisotropic electrostatics energy (Sec. 1.11.1)
    E_OFX: Tensor | None = None  # onsite Fock exchange correction energy (Sec. 1.16)
    E_MFX: Tensor | None = None  # long-range MFX exchange energy (doc/theory/20)
    E_ACP: Tensor | None = None  # atomic correction potentials energy (doc/theory/13)
    S: Tensor | None = None      # final overlap S^{sc} used for orthogonalization (doc/theory/5, Eq. 12)


def lowdin_orthogonalization(S: Tensor, eps_thresh: float = 1e-10) -> Tensor:
    evals, evecs = torch.linalg.eigh(S)
    # Guard: S must be symmetric positive-definite for Löwdin (doc/theory/5, Eq. 12)
    if torch.any(evals <= 0):
        min_eval = float(evals.min().item())
        raise ValueError(
            f"Overlap matrix S is not SPD (min eigenvalue={min_eval:.3e}). "
            "Check basis normalization and diatomic scaling to ensure S ≻ 0."
        )
    d = torch.clamp(evals, min=eps_thresh).rsqrt()
    # Scale eigenvectors' columns by d instead of forming a dense diagonal matrix
    return (evecs * d) @ evecs.T


def mulliken_charges(P: Tensor, S: Tensor, ao_atoms: Tensor) -> Tensor:
    PS = P @ S
    occ_diag = torch.diag(PS)
    nat = int(ao_atoms.max().item()) + 1
    # Vectorized accumulation of AO populations onto atoms
    return torch.bincount(ao_atoms.long(), weights=occ_diag, minlength=nat)


def _valence_electron_counts(numbers: Tensor, basis) -> Tensor:
    """Return per-atom valence electron counts consistent with shells present in basis.

    This implements the reference state for Δq in doc/theory/3 (Eq. 5): neutral valence
    electron counts per atom. We use the conservative shell counts from
    gxtb.hamiltonian.second_order_tb._electron_configuration_valence_counts and include
    only shells present for the element in the current basis (s,p,d,f).

    Returns tensor (nat,) on the same device/dtype as numbers (cast to float).
    """
    from .hamiltonian.second_order_tb import _electron_configuration_valence_counts as _valence_map
    device = numbers.device
    dtype = torch.get_default_dtype()
    # Build set of shells present per element in this basis
    elem_to_shells: Dict[int, set[str]] = {}
    for sh in basis.shells:
        elem_to_shells.setdefault(int(sh.element), set()).add(sh.l)
    out = torch.zeros(len(numbers), dtype=dtype, device=device)
    for i, zt in enumerate(numbers.tolist()):
        z = int(zt)
        present = elem_to_shells.get(z, set())
        val = _valence_map(z)
        out[i] = float(sum(v for l, v in val.items() if l in present))
    return out


def scf(
    numbers: Tensor,
    positions: Tensor,
    basis,
    build_h_core: Callable[[Tensor, Tensor, Dict[str, Tensor]], Dict[str, Tensor]],
    S: Tensor,
    hubbard: Dict[str, Tensor],
    ao_atoms: Tensor,
    nelec: int,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    mix: float = 0.5,
    mixing: Optional[Dict[str, object]] = None,  # {'scheme':'linear'|'anderson','beta':float,'history':int}
    etol: float | None = None,
    q_rms_tol: float | None = None,
    eeq_charges: Optional[Tensor] = None,
    second_order: bool = False,
    so_params: Optional[Dict[str, Tensor]] = None,
    uhf: bool = False,
    nelec_alpha: Optional[int] = None,
    nelec_beta: Optional[int] = None,
    fourth_order: bool = False,
    fourth_params: Optional[Dict[str, float]] = None,
    third_order: bool = False,
    third_shell_params: Optional[Dict[str, Tensor]] = None,  # expects {'shell_params': ShellSecondOrderParams, 'cn': Tensor}
    third_params: Optional[Dict[str, Tensor]] = None,  # expects dict packing ThirdOrderParams tensors
    mfx: bool = False,
    mfx_params: Optional[Dict[str, Tensor]] = None,  # expects {'alpha','omega','k1','k2','U_shell','xi_l','R0'(optional)}
    spin: bool = False,
    spin_params: Optional[Dict[str, Tensor]] = None,  # expects {'kW': (Zmax+1,), 'W0': (4,4)}
    # AES anisotropic electrostatics controls (doc/theory/16, Eqs. 109–111, 110a–b)
    aes: bool = False,
    aes_params: Optional[Dict[str, object]] = None,  # expects {'params': AESParams, 'r_cov': Tensor, 'k_cn': float}
    # OFX onsite exchange correction (doc/theory/21, Eqs. 155,159)
    ofx: bool = False,
    ofx_params: Optional[Dict[str, object]] = None,  # expects {'alpha': float, 'Lambda0_ao': Tensor}
    # Dispersion Fock (revD4/D4S): Eq. 174
    dispersion: bool = False,
    dispersion_params: Optional[Dict[str, object]] = None,  # expects {'mode': 'd4s'|'revD4','method': D4Method,'beta2': Tensor,'ref': dict,...}
    # ACP (doc/theory/13) non-local projectors
    acp: bool = False,
    acp_params: Optional[Dict[str, object]] = None,  # {'c0':(Zmax+1,4),'xi':(Zmax+1,4),'k_acp_cn': float,'cn_avg': Tensor,'r_cov': Tensor,'k_cn': float,'l_list': tuple[str,...](optional)}
    # Reference state charges for Δq (doc/theory/3, Eq. 5). If None: neutral (zeros).
    q_reference: Optional[Tensor] = None,
    # q‑vSZP dynamic overlap wiring (doc/theory/7 Eqs. 27–28; doc/theory/8 Eqs. 31–32)
    dynamic_overlap: bool = False,
    qvszp_params: Optional[Dict[str, object]] = None,  # expects {'k0','k1','k2','k3': (Zmax+1,), 'r_cov': Tensor, 'k_cn': float}
    # Numerical safety knobs
    spd_floor: float = 1e-8,  # minimum eigenvalue for S SPD projection used in Löwdin and Mulliken
) -> SCFResult:
    """Iterative SCF updating Hubbard onsite shifts until Mulliken charges converge."""
    nao = S.shape[0]
    # Guard: requested electrons cannot exceed basis capacity
    max_elec = 2 * nao if not uhf else 2 * nao  # same cap; split into spins later
    if nelec > max_elec:
        raise ValueError(
            f"Electron count nelec={nelec} exceeds AO capacity 2*nao={max_elec}. "
            "Use a larger basis or adjust valence electron estimation."
        )
    # S^{sc} used each iteration; start with provided matrix, rebuild if dynamic_overlap is enabled
    S_current = S
    # q represents Δq (charge fluctuations) per doc/theory/3, Eq. 5 (neutral reference by default)
    q = eeq_charges.clone() if eeq_charges is not None else torch.zeros(len(numbers), dtype=S.dtype, device=S.device)
    # Reference charges per shift-of-reference-state (doc/theory/3, Eq. 5): default neutral atoms (zeros)
    if q_reference is not None:
        if q_reference.shape != q.shape:
            raise ValueError("q_reference must match number of atoms (doc/theory/3, Eq. 5)")
        q_ref = q_reference.to(device=S.device, dtype=S.dtype)
    else:
        q_ref = torch.zeros_like(q)
    q_prev = q.clone()
    gamma = hubbard.get("gamma")
    gamma3 = hubbard.get("gamma3")
    P = torch.zeros((nao, nao), dtype=S.dtype, device=S.device)
    P_a = torch.zeros_like(P) if uhf else None
    P_b = torch.zeros_like(P) if uhf else None
    C = torch.zeros((nao, nao), dtype=S.dtype, device=S.device)
    eps_orb = torch.zeros(nao, dtype=S.dtype, device=S.device)
    H = torch.zeros_like(S)
    converged = False
    E_history: list[float] = []
    it = 0
    E2_current: Optional[Tensor] = None
    E3_current: Optional[Tensor] = None
    # Shell second-order precompute structures if requested
    shell_mode = False
    ao_shell: Optional[Tensor] = None
    shell_params_obj = None
    shell_gamma: Optional[Tensor] = None
    shell_q = None  # current shell charges (q_{l_A}, Eq. 99)
    shell_q_ref = None  # reference shell charges (initially zero since subtract p^0)
    ref_shell_pops = None  # p^0_{l_A} Eq. 99
    cn_tensor: Optional[Tensor] = None
    if second_order and so_params is not None and 'shell_params' in so_params:
        from .hamiltonian.second_order_tb import (
            ShellSecondOrderParams,
            compute_shell_charges,
            compute_gamma2_shell,
            compute_reference_shell_populations,
        )
        shell_params_obj = so_params['shell_params']  # type: ignore
        # Build AO -> shell mapping
        shells = basis.shells
        ao_shell_list = [0]*basis.nao
        for ish, off in enumerate(basis.ao_offsets):
            n_ao = basis.ao_counts[ish]
            for k in range(n_ao):
                ao_shell_list[off+k] = ish
        ao_shell = torch.tensor(ao_shell_list, dtype=torch.long, device=S.device)
        shell_mode = True
        # Coordination numbers if provided (either direct or via r_cov,k_cn)
        if 'cn' in so_params:
            cn_tensor = so_params['cn'].to(S.device)
        elif 'r_cov' in so_params and 'k_cn' in so_params:
            from .cn import coordination_number
            cn_tensor = coordination_number(positions, numbers, so_params['r_cov'].to(S.device), float(so_params['k_cn']))
        else:
            cn_tensor = torch.zeros(len(numbers), dtype=S.dtype, device=S.device)
        # Reference shell populations p^0_{l_A} (Eq. 99) & initial charges
        # Basis-aware p^0 option: drop contributions for valence shells not present in the basis
        if so_params.get('basis_aware_p0', False):
            from .hamiltonian.second_order_tb import compute_reference_shell_populations_basis_aware as _p0_ba
            ref_shell_pops = _p0_ba(numbers, basis).to(S.device, dtype=S.dtype)
        else:
            ref_shell_pops = compute_reference_shell_populations(numbers, basis).to(S.device, dtype=S.dtype)
        shell_q = torch.zeros(len(shells), dtype=S.dtype, device=S.device)
        shell_q_ref = torch.zeros_like(shell_q)
    dq_shell_current = None
    V_shell_current = None
    # Third-order precompute
    third_mode = False
    tau3 = None
    if third_order and third_shell_params is not None and third_params is not None:
        from .hamiltonian.third_order import compute_tau3_matrix, build_third_order_potentials, add_third_order_fock, ThirdOrderParams as _ThirdParams
        # Expect provided dicts
        sp = third_shell_params['shell_params']  # type: ignore
        cn_tensor3 = third_shell_params.get('cn', torch.zeros(len(numbers), dtype=S.dtype, device=S.device))  # type: ignore
        tp = third_params  # dict-like with keys: 'gamma3_elem','kGamma','k3','k3x'
        # Build per-shell U^{(2)} as in compute_gamma2_shell
        shells = basis.shells
        atom_idx_list = [sh.atom_index for sh in shells]
        z_list = torch.tensor([sh.element for sh in shells], dtype=torch.long, device=S.device)
        l_map = {'s':0,'p':1,'d':2,'f':3}
        l_idx = torch.tensor([l_map[sh.l] for sh in shells], dtype=torch.long, device=S.device)
        U0_shell = sp.U0[z_list, l_idx]
        kU_shell = sp.kU[z_list]
        cn3 = cn_tensor3.to(S.device)
        U_shell = U0_shell * (1.0 + kU_shell * cn3[torch.tensor(atom_idx_list, dtype=torch.long, device=S.device)])
        # Validate third-order inputs; disable third-order if invalid (no approximation applied)
        third_mode = False
        if torch.isfinite(U_shell).all() and (U_shell > 0).all():
            third_mode = True
        # Construct ThirdOrderParams object
        tparams = _ThirdParams(
            gamma3_elem=tp['gamma3_elem'].to(S.device),
            kGamma_l=(float(tp['kGamma'][0]), float(tp['kGamma'][1]), float(tp['kGamma'][2]), float(tp['kGamma'][3])),
            k3=float(tp['k3']),
            k3x=float(tp['k3x']),
        )
        if not (tparams.k3 > 0.0 and tparams.k3x > 0.0):
            third_mode = False
        if third_mode:
            tau3 = compute_tau3_matrix(numbers, positions, basis, U_shell, tparams)
            if not torch.isfinite(tau3).all():
                tau3 = None
                third_mode = False
        # Prepare reference shell populations for third-order shell charges q_{l_A} (Eq. 99)
        if so_params.get('basis_aware_p0', False):
            from .hamiltonian.second_order_tb import compute_reference_shell_populations_basis_aware as _ref_shell
        else:
            from .hamiltonian.second_order_tb import compute_reference_shell_populations as _ref_shell
        third_ref_shell_pops = _ref_shell(numbers, basis).to(S.device, dtype=S.dtype)
        third_shell_q = None  # updated after first density build
    # MFX precompute γ^{MFX} AO matrix
    gamma_mfx = None
    if mfx and mfx_params is not None:
        from .hamiltonian.mfx import MFXParams as _MFXParams, build_gamma_ao
        required = ['alpha','omega','k1','k2','U_shell','xi_l']
        for key in required:
            if key not in mfx_params:
                raise ValueError(f"mfx_params missing '{key}'")
        p = _MFXParams(alpha=float(mfx_params['alpha']), omega=float(mfx_params['omega']),
                       k1=float(mfx_params['k1']), k2=float(mfx_params['k2']),
                       U_shell=mfx_params['U_shell'], xi_l=mfx_params['xi_l'],
                       R0=mfx_params.get('R0'))
        gamma_mfx = build_gamma_ao(numbers, positions, basis, p)
    # Spin polarization (Eq. 124) setup
    use_spin = False
    kW = None
    W0 = None
    if spin and uhf:
        if spin_params is None or 'kW' not in spin_params or 'W0' not in spin_params:
            raise ValueError("spin=True requires spin_params with 'kW' and 'W0'")
        kW = spin_params['kW'].to(S.device)
        W0 = spin_params['W0'].to(S.device)
        use_spin = True
    # Precompute AO -> shell, AO -> atom, shell meta for spin
    ao_shell_idx = None
    ao_atom_idx = None
    shell_atom_idx = None
    shell_l_idx = None
    if use_spin:
        lmap = {'s':0,'p':1,'d':2,'f':3}
        # AO maps
        aosh = []
        aoat = []
        for ish, off in enumerate(basis.ao_offsets):
            n_ao = basis.ao_counts[ish]
            for k in range(n_ao):
                aosh.append(ish)
                aoat.append(basis.shells[ish].atom_index)
        ao_shell_idx = torch.tensor(aosh, dtype=torch.long, device=S.device)
        ao_atom_idx = torch.tensor(aoat, dtype=torch.long, device=S.device)
        # Shell meta
        shell_atom_idx = torch.tensor([sh.atom_index for sh in basis.shells], dtype=torch.long, device=S.device)
        shell_l_idx = torch.tensor([lmap[sh.l] for sh in basis.shells], dtype=torch.long, device=S.device)
    # Previous iteration shell magnetizations (m_l per shell)
    m_shell_prev = torch.zeros(len(basis.shells), dtype=S.dtype, device=S.device) if use_spin else None

    # Reference valence electrons for Δq (doc/theory/3): neutral valence counts per atom
    valence_e = _valence_electron_counts(numbers, basis).to(device=S.device, dtype=S.dtype)

    # AES precompute: build AO moment matrices once per geometry (Eq. 111a–c)
    use_aes = False
    S_mono = None
    Dm = None
    Qm = None
    aes_param_obj = None
    aes_r_cov = None
    aes_k_cn = None
    if aes:
        if aes_params is None or 'params' not in aes_params or 'r_cov' not in aes_params or 'k_cn' not in aes_params:
            raise ValueError("aes=True requires aes_params with {'params': AESParams, 'r_cov': Tensor, 'k_cn': float}")
        from .hamiltonian.moments_builder import build_moment_matrices
        S_mono, Dm, Qm = build_moment_matrices(numbers, positions, basis)
        aes_param_obj = aes_params['params']  # AESParams instance
        aes_r_cov = aes_params['r_cov'].to(S.device, dtype=S.dtype)
        aes_k_cn = float(aes_params['k_cn'])
        use_aes = True
    # OFX setup
    use_ofx = False
    ofx_param_obj = None
    if ofx:
        if ofx_params is None or 'alpha' not in ofx_params or 'Lambda0_ao' not in ofx_params:
            raise ValueError("ofx=True requires ofx_params with {'alpha': float, 'Lambda0_ao': Tensor}")
        from .hamiltonian.ofx import OFXParams as _OFXParams
        ofx_param_obj = _OFXParams(alpha=float(ofx_params['alpha']), Lambda0_ao=ofx_params['Lambda0_ao'])
        use_ofx = True
    # Dispersion setup (before iterations)
    use_disp = False
    disp_mode = None
    disp_method = None
    disp_beta2 = None
    disp_ref = None
    disp_d4s = None
    disp_cn_input = None
    disp_r_cov = None
    disp_k_cn = None
    if dispersion:
        if dispersion_params is None:
            raise ValueError("dispersion=True requires dispersion_params")
        if 'mode' not in dispersion_params or 'method' not in dispersion_params:
            raise ValueError("dispersion_params must contain 'mode' ('d4s'|'revD4') and 'method' (D4Method)")
        disp_mode = str(dispersion_params['mode']).lower()
        disp_method = dispersion_params['method']
        if disp_mode == 'd4s':
            required = ['beta2','ref','d4s_data']
            for k in required:
                if k not in dispersion_params:
                    raise ValueError(f"dispersion_params missing '{k}' for D4S mode")
            disp_beta2 = dispersion_params['beta2']
            disp_ref = dispersion_params['ref']
            disp_d4s = dispersion_params['d4s_data']
            # CN source
            if 'cn' in dispersion_params:
                disp_cn_input = dispersion_params['cn']
            else:
                if 'r_cov' not in dispersion_params or 'k_cn' not in dispersion_params:
                    raise ValueError("D4S mode requires 'cn' or both 'r_cov' and 'k_cn'")
                disp_r_cov = dispersion_params['r_cov']
                disp_k_cn = float(dispersion_params['k_cn'])
            use_disp = True
        elif disp_mode == 'revd4':
            raise ValueError("revD4 ζ Fock path not yet wired; provide D4S mode or A/B/C/D data")
        else:
            raise ValueError("dispersion_params['mode'] must be 'd4s' or 'revD4'")

    # ACP precompute: build S^ACP and H^ACP once per geometry (doc/theory/13, Eqs. 76–79)
    H_acp = None
    if acp:
        if acp_params is None:
            raise ValueError("acp=True requires acp_params")
        required = ['c0','xi','k_acp_cn','cn_avg','r_cov','k_cn']
        for rk in required:
            if rk not in acp_params:
                raise ValueError(f"acp_params missing '{rk}' (doc/theory/13, Eq. 80)")
        from .hamiltonian.acp import build_acp_overlap, acp_hamiltonian
        S_acp = build_acp_overlap(
            numbers, positions, basis,
            c0=acp_params['c0'], xi=acp_params['xi'],
            k_acp_cn=float(acp_params['k_acp_cn']), cn_avg=acp_params['cn_avg'],
            r_cov=acp_params['r_cov'], k_cn=float(acp_params['k_cn']),
            l_list=acp_params.get('l_list', ("s","p","d"))
        )
        H_acp = acp_hamiltonian(S_acp)

    # Mixing setup and mixer implementation (Anderson/Broyden/linear)
    # This mixer seeks a fixed point of q = F(q), where F(q) = q_Mulliken(q, P(S,H(q))) (doc/theory/5, Eq. 12 context)
    scheme = 'linear'
    beta = mix
    hist = 5
    # Adaptive damping knobs (no silent defaults: overridable via mixing dict)
    beta_min = 0.05
    beta_max = 0.8
    beta_decay = 0.5
    restart_on_nan = True
    if mixing is not None:
        scheme = str(mixing.get('scheme', 'linear')).lower()
        beta = float(mixing.get('beta', mix))
        hist = int(mixing.get('history', 5))
        beta_min = float(mixing.get('beta_min', beta_min))
        beta_max = float(mixing.get('beta_max', beta_max))
        beta_decay = float(mixing.get('beta_decay', beta_decay))
        restart_on_nan = bool(mixing.get('restart_on_nan', restart_on_nan))

    class _Mixer:
        """Simple mixer supporting 'linear', 'anderson', and 'broyden' on atomic charges.

        - Linear: q_{k+1} = (1-β) q_k + β F(q_k)
        - Anderson (Type-I): use residual r_k = F(q_k) - q_k, solve ΔR c ≈ r_k, then
              q_{k+1} = q_k + β r_k - (ΔX + β ΔR) c
        - Broyden (good): same linear system but use only ΔX, i.e.,
              q_{k+1} = q_k + β r_k - ΔX c
        All operations are torch-native and deterministic. Fallback to linear if history is singular.
        """
        def __init__(self, n: int, scheme: str, beta: float, hist: int):
            self.scheme = scheme
            self.beta = beta
            self.hist = max(1, int(hist))
            self.f_hist: list[Tensor] = []  # residual history r_i
            self.x_hist: list[Tensor] = []  # iterate history q_i

        def update(self, q_current: Tensor, q_mulliken: Tensor) -> Tensor:
            if self.scheme not in ('anderson', 'broyden'):
                return (1.0 - self.beta) * q_current + self.beta * q_mulliken
            r = (q_mulliken - q_current).detach().clone()
            x = q_current.detach().clone()
            self.f_hist.append(r)
            self.x_hist.append(x)
            if len(self.f_hist) < 2:
                return q_current + self.beta * r
            k = min(self.hist, len(self.f_hist) - 1)
            # Build ΔR and ΔX with the last k steps
            dR_cols = []
            dX_cols = []
            for i in range(-k, 0):
                dR_cols.append((self.f_hist[i] - self.f_hist[i - 1]).unsqueeze(1))
                dX_cols.append((self.x_hist[i] - self.x_hist[i - 1]).unsqueeze(1))
            dR = torch.cat(dR_cols, dim=1)
            dX = torch.cat(dX_cols, dim=1)
            # Solve least-squares for coefficients c: ΔR c ≈ r
            try:
                c = torch.linalg.lstsq(dR, r).solution
                if self.scheme == 'anderson':
                    correction = (dX + self.beta * dR) @ c
                else:  # broyden (good)
                    correction = dX @ c
                q_next = q_current + self.beta * r - correction
                return q_next
            except Exception:
                # Singular or ill-conditioned history; fallback to linear step
                return (1.0 - self.beta) * q_current + self.beta * q_mulliken

    mixer = _Mixer(len(numbers), scheme, beta, hist)

    def mix_update(q_current: Tensor, q_mulliken: Tensor) -> Tensor:
        return mixer.update(q_current, q_mulliken)

    def robust_eigh_sym(M: Tensor) -> tuple[Tensor, Tensor]:
        """Robust symmetric eigensolver with diagonal ridge and final general eig fallback.

        Returns (evals, evecs) with evals ascending and evecs column-orthonormal.
        """
        A = 0.5 * (M + M.T)
        ridge_vals = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
        for i, r in enumerate(ridge_vals + [None]):
            try:
                if r is not None:
                    A_reg = A + r * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
                else:
                    A_reg = A
                evals, evecs = torch.linalg.eigh(A_reg)
                return evals, evecs
            except Exception:
                if i < len(ridge_vals) - 1:
                    continue
                # Final fallback: general eigendecomposition + QR re-orthonormalization
                w, V = torch.linalg.eig(A)
                w = w.real
                V = V.real
                # Orthonormalize V deterministically
                Q, _ = torch.linalg.qr(V)
                # Rayleigh-Ritz to recover approximate evals under Q
                B = Q.T @ A @ Q
                d, _ = torch.linalg.eigh(0.5 * (B + B.T))
                return d, Q

    # Helper: validate qvszp_params pack deterministically
    def _validate_qv_pack(pack: Dict[str, object]) -> None:
        required = ("k0","k1","k2","k3","r_cov","k_cn")
        for rk in required:
            if rk not in pack:
                raise ValueError(f"qvszp_params missing '{rk}' (doc/theory/7 Eq. 28)")
        maxz = int(numbers.max().item())
        for kk in ("k0","k1","k2","k3"):
            v = pack[kk]
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"qvszp_params['{kk}'] must be a tensor")
            if v.dim() != 1 or v.shape[0] <= maxz:
                raise ValueError(f"qvszp_params['{kk}'] length must exceed max atomic number in system ({maxz})")
            if not torch.isfinite(v).all():
                raise ValueError(f"qvszp_params['{kk}'] contains non-finite entries")
        rcv = pack['r_cov']
        if not isinstance(rcv, torch.Tensor) or rcv.dim() != 1 or rcv.shape[0] <= maxz:
            raise ValueError("qvszp_params['r_cov'] must be a 1D tensor with per-element covalent radii covering all Z")
        # Allow zeros; CN code clamps denominators internally. Require non-negative and finite.
        if not torch.isfinite(rcv).all() or (rcv < 0).any():
            raise ValueError("qvszp_params['r_cov'] must be finite and non-negative")
        kcn = float(pack['k_cn'])
        if not (kcn > 0):
            raise ValueError("qvszp_params['k_cn'] must be positive")

    last_dq_rms = None
    for it in range(1, max_iter + 1):
        # Rebuild S and H0 if q‑vSZP dynamic overlap is enabled (Eq. 27–28 + Eqs. 31–32)
        ctx: Dict[str, Tensor] = {"q": q}
        if dynamic_overlap:
            if qvszp_params is None:
                raise ValueError("dynamic_overlap=True requires qvszp_params with keys {'k0','k1','k2','k3','r_cov','k_cn'} (doc/theory/7 Eq. 28)")
            _validate_qv_pack(qvszp_params)
            if eeq_charges is None:
                raise ValueError("dynamic_overlap=True requires eeq_charges to compute q^{EEQBC} (doc/theory/7 Eq. 28)")
            from .basis.qvszp import compute_effective_charge, build_dynamic_primitive_coeffs
            # Compute q_eff per Eq. (28)
            q_eff = compute_effective_charge(
                numbers,
                positions,
                q_scf=q,
                q_eeqbc=eeq_charges,
                r_cov=qvszp_params['r_cov'],
                k_cn=float(qvszp_params['k_cn']),
                k0=qvszp_params['k0'],
                k1=qvszp_params['k1'],
                k2=qvszp_params['k2'],
                k3=qvszp_params['k3'],
            )
            coeff_list = build_dynamic_primitive_coeffs(numbers, basis, q_eff)
            coeffs_map: Dict[int, Tensor] = {i: c for i, c in enumerate(coeff_list)}
            # Provide dynamic coefficients to core builder; it will compute S^{sc} consistently and return it
            ctx["coeffs"] = coeffs_map  # type: ignore[assignment]
        core = build_h_core(numbers, positions, ctx)
        H0 = core["H0"]
        # Prefer raw S for orthogonalization per doc/theory/5; builder may return both
        if 'S_raw' in core:
            S_current = core['S_raw']
        # Robust SPD projection for S before Löwdin/Mulliken to ensure numerical stability
        # Project S to symmetric positive-definite: S_spd = V diag(max(eig, spd_floor)) V^T
        Ssym = 0.5 * (S_current + S_current.T)
        evals, evecs = torch.linalg.eigh(Ssym)
        evals_clamped = torch.clamp(evals, min=float(spd_floor))
        S_current = (evecs * evals_clamped) @ evecs.T
        # Löwdin orthogonalization X = S^{-1/2} (doc/theory/5, Eq. 12 context)
        X = (evecs * evals_clamped.rsqrt()) @ evecs.T
        shift_atom = gamma[numbers.long()] * q
        if gamma3 is not None:
            shift_atom = shift_atom + gamma3[numbers.long()] * q.pow(3)
        shift_ao = shift_atom[ao_atoms.long()]
        H = H0.clone()
        H.diagonal().add_(shift_ao)
        # Fourth-order onsite Fock (Eq. 143)
        E4_current = None
        if fourth_order:
            if fourth_params is None or 'gamma4' not in fourth_params:
                raise ValueError("fourth_order enabled but missing 'gamma4' in fourth_params (Eq. 140b/143)")
            from .hamiltonian.fourth_order import add_fourth_order_fock, FourthOrderParams, fourth_order_energy
            dq_atom = q - q_ref
            fo = FourthOrderParams(gamma4=float(fourth_params['gamma4']))
            add_fourth_order_fock(H, S_current, ao_atoms, dq_atom, fo)
            E4_current = fourth_order_energy(dq_atom, fo)
        # Second-order shell-resolved or legacy atomic path
        if second_order and so_params is not None:
            if shell_mode and shell_params_obj is not None:
                from .hamiltonian.second_order_tb import (
                    compute_shell_charges,
                    compute_gamma2_shell,
                )
                # Compute gamma only once (no geometry update inside SCF here)
                if shell_gamma is None:
                    shell_gamma = compute_gamma2_shell(numbers, positions, basis, shell_params_obj, cn_tensor)
                # Use previous shell_q (from previous density) to build Fock contribution
                if shell_q is not None and ao_shell is not None and ref_shell_pops is not None:
                    dq_shell = shell_q - shell_q_ref  # Eq. 100a variable
                    V_shell = shell_gamma @ dq_shell  # Eq. 106
                    # Add F^{(2)} (Eq.105b) shell mapping
                    V_A = V_shell[ao_shell].unsqueeze(1)
                    V_B = V_shell[ao_shell].unsqueeze(0)
                    H.add_(0.5 * (V_A + V_B) * S_current)
                    # Energy contribution
                    E2_current = 0.5 * torch.einsum('i,ij,j->', dq_shell, shell_gamma, dq_shell)  # Eq. 100b
                    dq_shell_current = dq_shell
                    V_shell_current = V_shell
            else:
                from .hamiltonian.second_order_tb import (
                    compute_gamma2,
                    second_order_shifts,
                    add_second_order_fock,
                    SecondOrderParams,
                    second_order_energy,
                )
                params = SecondOrderParams(eta=so_params['eta'], r_cov=so_params['r_cov'])
                gamma2 = compute_gamma2(numbers, positions, params, device=H.device, dtype=H.dtype)
                shifts2 = second_order_shifts(gamma2, q, q_ref)
                # Atomic second-order Fock mapping (Eq. 105b):
                # F^{(2)}_{μν} = 1/2 S_{μν} ( V_A + V_B ), with V = γ^{(2)} Δq (Eq. 103)
                add_second_order_fock(H, S_current, ao_atoms, shifts2)
                E2_current = second_order_energy(gamma2, q, q_ref)
        # AES Fock update before diagonalization (Eq. 109 with 110a–b)
        E_AES_current = None
        if use_aes and S_mono is not None and Dm is not None and Qm is not None:
            from .hamiltonian.aes import aes_energy_and_fock  # eq: 109,110a,110b,111
            E_AES_current, H_AES = aes_energy_and_fock(
                numbers, positions, basis, P, S_mono, Dm, Qm, aes_param_obj, r_cov=aes_r_cov, k_cn=aes_k_cn,
                si_rules=aes_params.get('si_rules', None) if aes_params is not None else None
            )
            H = H + H_AES
        # OFX Fock update using current P (Eq. 159); energy per Eq. 155
        E_OFX_current = None
        if use_ofx and ofx_param_obj is not None:
            from .hamiltonian.ofx import add_ofx_fock, ofx_energy
            add_ofx_fock(H, numbers, basis, P, S_current, ofx_param_obj)
            E_OFX_current = ofx_energy(numbers, basis, P, S_current, ofx_param_obj)
        # MFX Fock (long-range exchange) per doc/theory/20
        E_MFX_current = None
        if gamma_mfx is not None:
            from .hamiltonian.mfx import mfx_fock, mfx_energy
            F_mfx = mfx_fock(P, S_current, gamma_mfx)
            H = H + F_mfx
            E_MFX_current = mfx_energy(P, F_mfx)
        # Dispersion Fock (Eq. 174) for D4S ζ path
        if use_disp and disp_mode == 'd4s':
            from .classical.dispersion import compute_d4s_atomic_potential
            if disp_cn_input is None:
                from .cn import coordination_number
                cn_vec = coordination_number(positions, numbers, disp_r_cov.to(S.device, dtype=S.dtype), disp_k_cn)
            else:
                cn_vec = disp_cn_input.to(S.device, dtype=S.dtype)
            V_atom = compute_d4s_atomic_potential(numbers, positions, cn_vec, q, disp_method, disp_beta2, disp_ref, disp_d4s)
            V_AO = V_atom[ao_atoms.long()].unsqueeze(1)
            H.add_(0.5 * (V_AO + V_AO.T) * S_current)

        # ACP contribution (density-independent) per Eq. 78c
        if H_acp is not None:
            H = H + H_acp

        # Third-order Fock (Eq. 136): use previous shell charges and current q
        if third_mode and tau3 is not None and 'third_shell_q' in locals() and third_shell_q is not None:
            V_shell3, V_atom3 = build_third_order_potentials(numbers, basis, third_shell_q, q, tau3, tparams)
            add_third_order_fock(H, S_current, basis, V_shell3, V_atom3)

        if not uhf:
            F = X.T @ H @ X
            F = 0.5 * (F + F.T)
            if not torch.isfinite(F).all():
                # Try large diagonal stabilization before bailing
                F = F + 1.0 * torch.eye(F.shape[0], dtype=F.dtype, device=F.device)
                if not torch.isfinite(F).all():
                    raise ValueError("Fock matrix contains non-finite entries; check parameter mappings (third/fourth-order, MFX/OFX)")
            eps_vals, C_orb = robust_eigh_sym(F)
            C = X @ C_orb
            nocc = nelec // 2
            P_new = 2.0 * C[:, :nocc] @ C[:, :nocc].T
        else:
            # UHF: same H for α/β unless spin terms added; separate occupations
            # Spin Fock: build F_spin from previous m_shell if enabled
            if use_spin:
                from .hamiltonian.spin import SpinParams as _SpinParams, add_spin_fock_uhf as _add_spin_fock_uhf
                spp = _SpinParams(kW_elem=kW, W0=W0)
                Ha = H.clone(); Hb = H.clone()
                _add_spin_fock_uhf(Ha, Hb, S_current, numbers, basis, P_a, P_b, spp)
            else:
                Ha = H
                Hb = H
            Fa = X.T @ Ha @ X
            Fa = 0.5 * (Fa + Fa.T)
            if not torch.isfinite(Fa).all():
                Fa = Fa + 1.0 * torch.eye(Fa.shape[0], dtype=Fa.dtype, device=Fa.device)
                if not torch.isfinite(Fa).all():
                    raise ValueError("Spin-α Fock contains non-finite entries; check parameters")
            # α
            eps_a, Ca_orb = robust_eigh_sym(Fa)
            Ca = X @ Ca_orb
            # β
            Fb = X.T @ Hb @ X
            Fb = 0.5 * (Fb + Fb.T)
            if not torch.isfinite(Fb).all():
                Fb = Fb + 1.0 * torch.eye(Fb.shape[0], dtype=Fb.dtype, device=Fb.device)
                if not torch.isfinite(Fb).all():
                    raise ValueError("Spin-β Fock contains non-finite entries; check parameters")
            eps_b, Cb_orb = robust_eigh_sym(Fb)
            Cb = X @ Cb_orb
            # occupations
            if nelec_alpha is None or nelec_beta is None:
                # default: split nelec across spins as even as possible
                na = (nelec + 1) // 2
                nb = nelec // 2
            else:
                na, nb = nelec_alpha, nelec_beta
            Pa_new = Ca[:, :na] @ Ca[:, :na].T
            Pb_new = Cb[:, :nb] @ Cb[:, :nb].T
            P_new = Pa_new + Pb_new
            P_a, P_b = Pa_new, Pb_new
            eps_vals = 0.5 * (eps_a + eps_b)
            C = 0.5 * (Ca + Cb)
            # Update shell magnetizations for next step
            if use_spin:
                from .hamiltonian.spin import compute_shell_magnetizations
                m_shell_prev = compute_shell_magnetizations(P_a, P_b, S, basis)
        # Charge fluctuations Δq = N_valence - N_Mulliken (doc/theory/3, Eq. 5)
        q_mull = valence_e - mulliken_charges(P_new, S_current, ao_atoms)
        q = mix_update(q, q_mull)
        # Update shell charges for next iteration (after density update)
        if shell_mode and shell_params_obj is not None and ref_shell_pops is not None:
            from .hamiltonian.second_order_tb import compute_shell_charges
            shell_q = compute_shell_charges(P_new, S, basis, ref_shell_pops)
        # Electronic energy (first-order expectation). Note that AES energy is accumulated separately
        # using the density used to form H^{AES} (Eqs. 109–111). Here we keep Tr(P H) history for convergence.
        E_el = torch.einsum('ij,ji->', P_new, H)
        # Adaptive mixing: handle non-finite energies by reducing beta and restarting charge update
        if not torch.isfinite(E_el):
            if restart_on_nan:
                beta = max(beta * beta_decay, beta_min)
                f_hist.clear(); x_hist.clear()
                # Back off to previous charges
                q = q_prev.clone()
                # Skip convergence checks; continue with reduced beta
                continue
            else:
                raise ValueError("Electronic energy non-finite during SCF; consider enabling restart_on_nan in mixing")
        # Optional ACP energy bookkeeping (Eq. 75c)
        E_ACP_current = None
        if H_acp is not None:
            E_ACP_current = torch.einsum('ij,ji->', P_new, H_acp)
        # Update third-order shell charges for next step
        if third_mode and 'third_ref_shell_pops' in locals():
            from .hamiltonian.second_order_tb import compute_shell_charges as _comp_shell_q
            third_shell_q = _comp_shell_q(P_new, S, basis, third_ref_shell_pops)
        # Accumulate third-order energy with current shell charges
        if third_mode and tau3 is not None:
            from .hamiltonian.third_order import third_order_energy
            if 'third_shell_q' in locals() and third_shell_q is not None:
                E3_current = third_order_energy(numbers, positions, basis, third_shell_q, q, U_shell, tparams)
        # Convergence checks (default: charge sup-norm; optional energy and RMS)
        e_val = float(E_el.item())
        E_history.append(e_val)
        dq = q - q_prev
        cond = torch.max(torch.abs(dq)) < tol
        if q_rms_tol is not None:
            dq_rms = torch.sqrt(torch.mean(dq * dq))
            cond = cond and (dq_rms < q_rms_tol)
            # Adapt beta based on RMS progression
            if last_dq_rms is not None and torch.isfinite(dq_rms):
                if dq_rms > last_dq_rms * 1.25:  # diverging
                    beta = max(beta * beta_decay, beta_min)
                    f_hist.clear(); x_hist.clear()
                elif dq_rms < last_dq_rms * 0.5 and beta < beta_max:  # good progress
                    beta = min(beta * (1.0 / max(beta_decay, 1e-6)), beta_max)
            last_dq_rms = dq_rms
        if etol is not None and len(E_history) >= 2:
            cond = cond and (abs(E_history[-1] - E_history[-2]) < etol)
        if cond:
            P = P_new
            eps_orb = eps_vals
            converged = True
            break
        P = P_new
        eps_orb = eps_vals
        q_prev = q.clone()
    E_final = torch.tensor(E_history[-1], dtype=S.dtype, device=S.device) if E_history else None
    return SCFResult(H=H, P=P, P_alpha=P_a, P_beta=P_b, q=q, q_ref=q_ref, eps=eps_orb, C=C, n_iter=it, converged=converged, E_elec=E_final,
                     E2=E2_current, E3=E3_current, E4=E4_current, E_history=E_history, dq_shell=dq_shell_current, V_shell=V_shell_current,
                     E_AES=E_AES_current, E_OFX=E_OFX_current, E_MFX=E_MFX_current, E_ACP=E_ACP_current, S=S_current)
