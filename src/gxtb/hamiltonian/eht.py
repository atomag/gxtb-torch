"""Extended Hückel (EHT) Hamiltonian orchestration (0th + 1st order TB part).

Referenced equations from theory docs:
    Energy trace form: E^{EHT} = Tr{ H^{EHT} ∘ P } (Eq. 63)
    Matrix elements (original arithmetic mean form): Eq. 64
    Shell CN dependence (fallback linear form): H_{l_A} = h_{l_A} - k^{H0,CN}_{l_A} * CN_A (Eq. 65)
    Shell pair distance polynomial factorisation: Π_{l_A l_B}(R) = π_{l_A}(R) π_{l_B}(R) (Eq. 66)
    Atomic shell distance polynomial (linear in R/R_cov): π_{l_A}(R) = 1 + k^{shp}_A k^{shp,l}_{l_A} (R / R^{cov}_{AB}) (Eq. 67)

Code deviations / extensions:
    - Optional geometric mean √(k_{l_A} k_{l_B}) vs arithmetic (k_{l_A}+k_{l_B})/2 Wolfsberg factor of Eq. 64 via wolfsberg_mode switch (default arithmetic = Eq. 64).
    - Optional electronegativity damping X_{AB} = max(0, 1 - k_en (ΔEN)^2) (not in original Eq. 64) to modulate hetero‑pairs.
    - Overlap S is pre-scaled in the diatomic frame (σ/π/δ) per earlier Sec. 1.3 equations (Eqs. 31–32) handled in build_scaled_overlap.
    - Onsite ε_{l_A} may use a higher-order CN polynomial (when available) instead of strictly linear Eq. 65.
    - f shells included structurally; overlap and rotations support up to f/g; no zero placeholders.

Limitations / TODO:
    - Analytical gradients: main terms (Eqs. 68–71) and diatomic-scaled-overlap term (Eq. 39) implemented.
    - Higher TB orders (Sec. 1.9+) and charge self-consistency handled elsewhere.
    - EN damping & geometric mean variants should be parameter-switch controlled later.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from ..basis.qvszp import AtomBasis
from ..params.loader import GxTBParameters
from ..params.schema import GxTBSchema, map_eht_params, map_cn_params
from .overlap_tb import build_overlap, build_scaled_overlap
from .onsite_tb import build_onsite
from .distance_tb import distance_factor, distance_factor_with_grad, en_penalty
from .utils_tb import geometric_mean


@dataclass
class EHTResult:
    S_raw: torch.Tensor      # Unscaled AO overlap (lab frame)
    S_scaled: torch.Tensor   # Diatomic-channel scaled overlap (Eq. 31–32 applied)
    H: torch.Tensor          # First-order EHT Hamiltonian
    eps: torch.Tensor        # Onsite orbital energies (per AO)
    energy_first_order: torch.Tensor | None = None  # Optional Tr(PH) (Eq. 63)


def build_eht_hamiltonian(
        numbers: torch.Tensor,
        positions: torch.Tensor,
        basis: AtomBasis,
        gparams: GxTBParameters,
        schema: GxTBSchema,
        r_cov: torch.Tensor | None = None,
        k_cn: float | None = None,
    wolfsberg_mode: str = "arithmetic",  # "arithmetic" (Eq.64) or "geometric" (design variant)
    *,
    S_raw_override: torch.Tensor | None = None,
    S_scaled_override: torch.Tensor | None = None,
) -> EHTResult:
    """Construct first-order (EHT) Hamiltonian.

    Steps:
        1. Raw overlap (S_raw)
        2. Diatomic channel scaling (S_scaled) – Eqs. 31–32
        3. Onsite ε_{lA} via CN polynomial or linear Eq. 65
        4. Off-diagonal H_{μν} (μ≠ν): geometric_mean(k) * avg(ε) * Π(R) * X_EN * S_scaled (mod. Eq. 64)

    Parameters
    ----------
    numbers : (nat,) torch.int64
    positions : (nat,3) float
    basis : AtomBasis
    gparams : parameter container
    schema : schema mapping definitions
    r_cov, k_cn : optional overrides for CN covalent radii / constant
    wolfsberg_mode : 'arithmetic' to use (k_A + k_B)/2 (Eq. 64), 'geometric' for √(k_A k_B)
    """
    # Overlap matrices (doc/theory/8, Eqs. 31–32). Allow external overrides to ensure
    # consistency with dynamic q‑vSZP contractions when provided by SCF (no shortcuts; same S used in H and SCF).
    S_raw = S_raw_override if S_raw_override is not None else build_overlap(numbers, positions, basis)
    S_scaled = S_scaled_override if S_scaled_override is not None else build_scaled_overlap(numbers, positions, basis, gparams, schema)
    cn_map = map_cn_params(gparams, schema) if schema.cn else None
    r_cov_eff = r_cov if r_cov is not None else (cn_map['r_cov'] if cn_map else None)
    k_cn_eff = k_cn if k_cn is not None else (cn_map['k_cn'] if cn_map else None)
    eps = build_onsite(numbers, positions, basis, gparams, schema, r_cov_eff, k_cn_eff)
    eht = map_eht_params(gparams, schema)
    # Inject r_cov for Eq. 67 normalization in Π(R) if available
    if r_cov_eff is None and cn_map is not None:
        r_cov_eff = cn_map['r_cov']
    if r_cov_eff is not None:
        eht = dict(eht)
        eht['r_cov'] = r_cov_eff.to(dtype=positions.dtype, device=positions.device)
    has_en = 'en' in eht and 'k_en' in eht
    en = eht['en'] if has_en else None
    k_en = float(eht['k_en'][0].item()) if has_en else None
    H = torch.zeros_like(S_scaled)
    H.diagonal().copy_(eps)
    dist = torch.cdist(positions, positions)
    for i, shi in enumerate(basis.shells):
        oi, ni = basis.ao_offsets[i], basis.ao_counts[i]
        ZA = shi.element
        lA = shi.l
        epsA = eps[oi:oi+ni].mean()
        kWA = eht.get(f'k_w_{lA}', eht.get('k_w_s', torch.ones_like(eps)))[ZA]
        for j, shj in enumerate(basis.shells):
            if j <= i:
                continue
            oj, nj = basis.ao_offsets[j], basis.ao_counts[j]
            ZB = shj.element
            lB = shj.l
            epsB = eps[oj:oj+nj].mean()
            kWB = eht.get(f'k_w_{lB}', eht.get('k_w_s', torch.ones_like(eps)))[ZB]
            if wolfsberg_mode == "geometric":
                kpair = geometric_mean(kWA, kWB)
            else:
                # default arithmetic Eq. 64
                kpair = 0.5 * (kWA + kWB)
            R_AB = dist[shi.atom_index, shj.atom_index]
            Pi_R = distance_factor(lA, lB, ZA, ZB, R_AB, eht)
            X = en_penalty(ZA, ZB, en, k_en)
            avg_eps = 0.5 * (epsA + epsB)
            blockS = S_scaled[oi:oi+ni, oj:oj+nj]
            Hblk = kpair * avg_eps * Pi_R * X * blockS
            H[oi:oi+ni, oj:oj+nj] = Hblk
            H[oj:oj+nj, oi:oi+ni] = Hblk.T
    return EHTResult(S_raw=S_raw, S_scaled=S_scaled, H=H, eps=eps)


def first_order_energy(P: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Compute E^{EHT} = Tr{ P H } (Eq. 63).

    Uses einsum for clarity. P and H assumed symmetric.
    """
    return torch.einsum('ij,ji->', P, H)

__all__ = ["EHTResult", "build_eht_hamiltonian", "first_order_energy"]
def eht_energy_gradient(
    numbers: torch.Tensor,
    positions: torch.Tensor,
    basis: AtomBasis,
    gparams: GxTBParameters,
    schema: GxTBSchema,
    P: torch.Tensor,
    *,
    wolfsberg_mode: str = "arithmetic",
    # Optional CN-driven dynamic-overlap contribution (doc/theory/7 Eq. 27–28 + doc/theory/8 Eq. 39)
    dynamic_overlap_cn: dict | None = None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Analytic gradient of E^{EHT} = Tr(P H^{EHT}).

    Includes:
      - ∂H_{l}/∂R via CN (Eq. 69) using dCN/dR (Eq. 48b)
      - ∂Π/∂R via distance polynomials (Eqs. 70–71)
      - ∂Ŝ^{sc}/∂R via diatomic rotation and explicit overlap dependence (Eq. 39)

    Notes:
      - The explicit ∂Ŝ^{sc}/∂R is formed by differentiating the contracted block
        Σ_{μν∈(i,j)} P_{μν} S^{sc}_{μν}(R) w.r.t. R_AB using autograd on the exact
        McMurchie–Davidson kernel and diatomic rotations (no approximations).
    """
    device = positions.device
    dtype = positions.dtype
    nat = numbers.shape[0]
    dE = torch.zeros((nat, 3), dtype=dtype, device=device)
    dE_eps = torch.zeros_like(dE)
    dE_pi = torch.zeros_like(dE)
    dE_s = torch.zeros_like(dE)
    dE_coeff = torch.zeros_like(dE)
    # Build onsite eps and prerequisites
    cn_map = map_cn_params(gparams, schema)
    r_cov = cn_map['r_cov'].to(device=device, dtype=dtype)
    k_cn = float(cn_map['k_cn'])
    # dCN/dR_X: (A, X, 3)
    from ..cn import coordination_number, coordination_number_grad
    cn = coordination_number(positions, numbers, r_cov, k_cn)
    dcn = coordination_number_grad(positions, numbers, r_cov, k_cn)  # (nat,nat,3)
    eht = map_eht_params(gparams, schema)
    # Inject r_cov for Eq. 67 normalization in Π(R)
    if 'r_cov' not in eht:
        try:
            cnm = map_cn_params(gparams, schema)
            eht = dict(eht)
            eht['r_cov'] = cnm['r_cov'].to(dtype=dtype, device=device)
        except Exception:
            pass
    # AO maps
    ao_off = basis.ao_offsets
    ao_cnt = basis.ao_counts
    ao_atoms = []
    for ish, off in enumerate(ao_off):
        ao_atoms.extend([basis.shells[ish].atom_index] * ao_cnt[ish])
    ao_atoms = torch.tensor(ao_atoms, dtype=torch.long, device=device)
    # Diagonal contribution: Σ_μ P_μμ dε_{A(μ)}
    diagP = torch.diag(P)
    # Per-atom weight wA = Σ_{μ∈A} P_μμ
    wA = torch.zeros(nat, dtype=dtype, device=device)
    for mu, A in enumerate(ao_atoms.tolist()):
        wA[A] += diagP[mu]
    # Helper: dε/ dR_X for a given shell (ish) -> (nat,3)
    from .utils_tb import poly_eval_derivative
    def d_eps_shell(ish: int) -> torch.Tensor:
        A = basis.shells[ish].atom_index
        z = basis.shells[ish].element
        l = basis.shells[ish].l
        poly_key = f"pi_cn_{l}"
        kho_key = f"k_ho_{l}"
        if poly_key in eht:
            coeff = eht[poly_key][z].to(device=device, dtype=dtype)
            dpoly = poly_eval_derivative(coeff, cn[A])
            return dpoly * dcn[A]
        elif kho_key in eht:
            k_ho = eht[kho_key][z].to(device=device, dtype=dtype)
            return (-k_ho) * dcn[A]
        else:
            return torch.zeros((nat, 3), dtype=dtype, device=device)
    # Per-shell weight w_shell = Σ_{μ∈shell} P_μμ
    w_shell = torch.zeros(len(basis.shells), dtype=dtype, device=device)
    for ish, sh in enumerate(basis.shells):
        off = ao_off[ish]; n = ao_cnt[ish]
        w_shell[ish] = diagP[off:off+n].sum()
    # Accumulate diagonal gradient
    for ish in range(len(basis.shells)):
        d_eps_mat = d_eps_shell(ish)  # (nat,3)
        scalar = float(w_shell[ish].item())
        dE_eps = dE_eps + scalar * d_eps_mat
    # Off-diagonal contribution from avg_eps and Π(R)
    dist = torch.cdist(positions, positions)
    # Precompute S_scaled to get block overlaps consistent with energy assembly
    S_scaled = build_scaled_overlap(numbers, positions, basis, gparams, schema)
    # Pre-extract EN + Wolfsberg params
    has_en = 'en' in eht and 'k_en' in eht
    en = eht['en'] if has_en else None
    k_en = float(eht['k_en'][0].item()) if has_en else None
    def kW_of(shell_l: str, Z: int) -> torch.Tensor:
        key = f'k_w_{shell_l}'
        if key in eht:
            return eht[key][Z].to(device=device, dtype=dtype)
        return torch.tensor(1.0, dtype=dtype, device=device)
    # Compute eps per shell scalar (same as used for avg_eps)
    # Reuse build_onsite to be consistent
    eps_ao = build_onsite(numbers, positions, basis, gparams, schema, r_cov, k_cn)
    # Helper to average eps within shell
    def shell_eps(ish: int) -> torch.Tensor:
        off = ao_off[ish]; n = ao_cnt[ish]
        return eps_ao[off:off+n].mean()
    for i, shi in enumerate(basis.shells):
        A = shi.atom_index; ZA = shi.element; lA = shi.l
        oi, ni = ao_off[i], ao_cnt[i]
        epsA = shell_eps(i)
        kWA = kW_of(lA, ZA)
        for j, shj in enumerate(basis.shells):
            if j <= i:
                continue
            B = shj.atom_index; ZB = shj.element; lB = shj.l
            oj, nj = ao_off[j], ao_cnt[j]
            epsB = shell_eps(j)
            kWB = kW_of(lB, ZB)
            if wolfsberg_mode == "geometric":
                kpair = torch.sqrt(torch.clamp(kWA, min=0.0) * torch.clamp(kWB, min=0.0))
            else:
                kpair = 0.5 * (kWA + kWB)
            R_AB = dist[A, B]
            Pi, dPi_dR = distance_factor_with_grad(lA, lB, ZA, ZB, R_AB, eht)
            Xpen = en_penalty(ZA, ZB, en, k_en)
            avg_eps = 0.5 * (epsA + epsB)
            # Block weights: sum(P ∘ S_scaled) over (i,j)
            Pblk = P[oi:oi+ni, oj:oj+nj]
            # If Pblk is (near) zero, skip to avoid 0*NaN propagation from S blocks
            if torch.all(Pblk.abs() < 1e-15):
                continue
            Sblk = S_scaled[oi:oi+ni, oj:oj+nj]
            mask_nz = Pblk.abs() > 0
            W = (Pblk[mask_nz] * Sblk[mask_nz]).sum()
            # d(avg_eps) term: vector contribution for every atom X
            dA = d_eps_shell(i)
            dB = d_eps_shell(j)
            s_pref = 2.0 * float((kpair * Xpen * Pi * W).item())
            for X in range(nat):
                d_avg = 0.5 * (dA[X] + dB[X])  # (3,)
                dE_eps[X] = dE_eps[X] + s_pref * d_avg
            # dΠ/dR term: along ±r_hat
            if R_AB.item() > 1e-14:
                rhat = (positions[A] - positions[B]) / R_AB
                scalar2 = 2.0 * float((kpair * Xpen * avg_eps * W * dPi_dR).item())
                dE_pi[A] = dE_pi[A] + scalar2 * rhat
                dE_pi[B] = dE_pi[B] - scalar2 * rhat
    # Note: Off-diagonal terms require S_block; for Step A tests with diagonal P, these terms vanish (P_block zeros).
    # --- Additional contribution: coefficient-induced overlap change via CN in q_eff (Eq. 27–28) ---
    # If dynamic_overlap_cn is provided, include the gradient path from c(q_eff(CN)) using Eq. 39 separation
    # restricted to the implicit S dependence through c (no SCF/EEQ response terms; see notes below).
    if dynamic_overlap_cn is not None:
        # Validate pack
        required_keys = ("k0", "k1", "k2", "k3", "r_cov", "k_cn", "q_scf", "q_eeqbc")
        for k in required_keys:
            if k not in dynamic_overlap_cn:
                raise ValueError(
                    f"dynamic_overlap_cn missing '{k}' (doc/theory/7_q-vSZP_basis_set.md Eq. 28)"
                )
        k0 = dynamic_overlap_cn["k0"].to(device=device, dtype=dtype)
        k1 = dynamic_overlap_cn["k1"].to(device=device, dtype=dtype)
        k2 = dynamic_overlap_cn["k2"].to(device=device, dtype=dtype)
        k3 = dynamic_overlap_cn["k3"].to(device=device, dtype=dtype)
        r_cov = dynamic_overlap_cn["r_cov"].to(device=device, dtype=dtype)
        k_cn = float(dynamic_overlap_cn["k_cn"])
        q_scf = dynamic_overlap_cn["q_scf"].to(device=device, dtype=dtype)
        q_eeq = dynamic_overlap_cn["q_eeqbc"].to(device=device, dtype=dtype)

        # CN and its derivative (doc/theory/9_cn.md Eq. 47, 48b)
        from ..cn import coordination_number, coordination_number_grad
        cn = coordination_number(positions, numbers, r_cov, k_cn)
        dcn = coordination_number_grad(positions, numbers, r_cov, k_cn)  # (nat,nat,3) A,X,vec
        # ∂q_eff/∂CN = k2 * (1/(2 sqrt(CN))) + k3 * q_EEQBC (Eq. 28); requires CN > 0
        if (cn <= 0).any():
            raise ValueError(
                "CN contains non-positive entries; ∂sqrt(CN)/∂CN undefined at CN<=0 (Eq. 28)."
            )
        Z = numbers.to(dtype=torch.long, device=device)
        dqeff_dcn = (0.5 * k2[Z] / torch.sqrt(cn)) + (k3[Z] * q_eeq)

        # Diatomic scaling parameters per element (Eqs. 31–32)
        from ..params.schema import map_diatomic_params
        diat = map_diatomic_params(gparams, schema) if schema.diatomic else None
        # Helper to build per-element channel dict expected by scale_diatomic_overlap
        def k_channels_for_Z(Zval: int) -> dict:
            if diat is None:
                return {}
            return {
                "sigma": float(diat["sigma"][Zval].item()),
                "pi": float(diat["pi"][Zval].item()),
                "delta": float(diat["delta"][Zval].item()),
            }

        # Overlap derivative blocks wrt q_eff(A) via primitive coefficient slopes c1 (Eq. 27)
        from ..basis.md_overlap import overlap_shell_pair
        from ..basis.overlap import scale_diatomic_overlap
        # Shell/meta
        shells = basis.shells
        n_shell = len(shells)
        ao_off = basis.ao_offsets
        ao_cnt = basis.ao_counts
        atom_idx = torch.tensor([sh.atom_index for sh in shells], dtype=torch.long, device=device)
        lmap = {"s": 0, "p": 1, "d": 2, "f": 3}
        l_idx = [lmap[sh.l] for sh in shells]
        # Precompute alpha, current coeff c, and slope c1 per shell
        alpha = [torch.tensor([p[0] for p in sh.primitives], dtype=dtype, device=device) for sh in shells]
        c0 = [torch.tensor([p[1] for p in sh.primitives], dtype=dtype, device=device) for sh in shells]
        c1v = [torch.tensor([p[2] for p in sh.primitives], dtype=dtype, device=device) for sh in shells]
        # Current q_eff (Eq. 28) to evaluate coefficients c = c0 + c1 * q_eff
        from ..basis.qvszp import compute_effective_charge
        q_eff = compute_effective_charge(numbers, positions, q_scf, q_eeq, r_cov=r_cov, k_cn=k_cn, k0=k0, k1=k1, k2=k2, k3=k3)
        coeff = [c0[i] + c1v[i] * q_eff[atom_idx[i]] for i in range(n_shell)]

        # Accumulate gradient vector dE/dR_X over atoms X (nat,3)
        dE_cn = torch.zeros((nat, 3), dtype=dtype, device=device)
        # Pairwise distances for Pi_R etc.
        dist = torch.cdist(positions, positions)
        # Pre-extract EN + Wolfsberg params (same as above helper functions)
        def kW_of(shell_l: str, Zint: int) -> torch.Tensor:
            key = f'k_w_{shell_l}'
            if key in eht:
                return eht[key][Zint].to(device=device, dtype=dtype)
            return torch.tensor(1.0, dtype=dtype, device=device)

        for i in range(n_shell):
            A = int(atom_idx[i].item()); ZA = int(numbers[A].item()); lA = shells[i].l
            oi, ni = ao_off[i], ao_cnt[i]
            epsA = shell_eps(i)
            kWA = kW_of(lA, ZA)
            for j in range(i + 1, n_shell):
                B = int(atom_idx[j].item()); ZB = int(numbers[B].item()); lB = shells[j].l
                oj, nj = ao_off[j], ao_cnt[j]
                epsB = shell_eps(j)
                kWB = kW_of(lB, ZB)
                if wolfsberg_mode == "geometric":
                    kpair = torch.sqrt(torch.clamp(kWA, min=0.0) * torch.clamp(kWB, min=0.0))
                else:
                    kpair = 0.5 * (kWA + kWB)
                R_AB = dist[A, B]
                Pi, _ = distance_factor_with_grad(lA, lB, ZA, ZB, R_AB, eht)
                Xpen = en_penalty(ZA, ZB, en, k_en)
                avg_eps = 0.5 * (epsA + epsB)
                # Construct derivative S_block wrt q_eff(A) and q_eff(B) at fixed R (explicit c dependence only)
                ai = alpha[i]; aj = alpha[j]
                ci = coeff[i]; cj = coeff[j]
                c1i = c1v[i]; c1j = c1v[j]
                R = positions[A] - positions[B]
                # Contributions: dS/dq_eff(A): overlap(c1_i, c_j) if shell i on A; plus overlap(c_i, c1_j) if shell j on A
                # Use spherical transforms computed from the physical contracted shells (ci,cj) to avoid
                # degenerate on-center metrics when c1 vectors are zero.
                from ..basis.md_overlap import _metric_transform_for_shell as _T_sph, _overlap_cart_block as _S_cart
                Ti = _T_sph(l_idx[i], ai, ci)
                Tj = _T_sph(l_idx[j], aj, cj)
                dS_A = torch.zeros((ni, nj), dtype=dtype, device=device)
                if atom_idx[i].item() == A:
                    Sc = _S_cart(l_idx[i], l_idx[j], ai, c1i, aj, cj, R)
                    dS_A = dS_A + (Ti @ Sc @ Tj.T)
                if atom_idx[j].item() == A:
                    Sc = _S_cart(l_idx[i], l_idx[j], ai, ci, aj, c1j, R)
                    dS_A = dS_A + (Ti @ Sc @ Tj.T)
                dS_B = torch.zeros((ni, nj), dtype=dtype, device=device)
                if atom_idx[i].item() == B:
                    Sc = _S_cart(l_idx[i], l_idx[j], ai, c1i, aj, cj, R)
                    dS_B = dS_B + (Ti @ Sc @ Tj.T)
                if atom_idx[j].item() == B:
                    Sc = _S_cart(l_idx[i], l_idx[j], ai, ci, aj, c1j, R)
                    dS_B = dS_B + (Ti @ Sc @ Tj.T)
                # Apply diatomic scaling (Eqs. 31–32) to derivative blocks (K independent of q_eff)
                kA_ch = k_channels_for_Z(ZA)
                kB_ch = k_channels_for_Z(ZB)
                dS_A_scaled = scale_diatomic_overlap(dS_A, R, lA, lB, kA_ch, kB_ch)
                dS_B_scaled = scale_diatomic_overlap(dS_B, R, lA, lB, kA_ch, kB_ch)
                # Contract with P block to form scalar sensitivities sA, sB
                Pblk = P[oi:oi+ni, oj:oj+nj]
                sA = (Pblk * dS_A_scaled).sum()
                sB = (Pblk * dS_B_scaled).sum()
                # Prefactor from H assembly (modified Eq. 64): 2 * kpair * Xpen * avg_eps * Pi
                pref = 2.0 * (kpair * Xpen * Pi * avg_eps)
                # Accumulate into dE via chain rule: dE += pref * sA * (dqeff/dCN)_A * dCN_A/dR_X + likewise for B
                # dCN has shape (A,X,3)
                dE_cn = dE_cn + pref * (sA * dqeff_dcn[A]) * dcn[A] + pref * (sB * dqeff_dcn[B]) * dcn[B]
        dE_coeff = dE_coeff + dE_cn
    # --- Step B: explicit ∂Ŝ^{sc}/∂R contribution (doc/theory/8 Eq. 39) ---
    # Autograd-based contraction per shell pair using differentiable overlap and rotations.
    from ..basis.md_overlap import overlap_shell_pair_torch as _ov_sph_torch
    from ..basis.overlap import scale_diatomic_overlap as _scale_s, rotation_matrices as _rot
    from ..params.schema import map_diatomic_params
    diat = map_diatomic_params(gparams, schema) if schema.diatomic else None
    lmap = {"s": 0, "p": 1, "d": 2, "f": 3}
    # Prepare per-shell primitive tensors and baseline coefficients (static c0)
    shells = basis.shells
    alpha = [torch.tensor([p[0] for p in sh.primitives], dtype=dtype, device=device) for sh in shells]
    c0 = [torch.tensor([p[1] for p in sh.primitives], dtype=dtype, device=device) for sh in shells]
    # If dynamic_overlap_cn present, optionally use q_eff for coefficients to mirror SCF overlap
    use_qeff_coeff = False
    q_eff_local = None
    if dynamic_overlap_cn is not None:
        try:
            k0 = dynamic_overlap_cn["k0"].to(device=device, dtype=dtype)
            k1 = dynamic_overlap_cn["k1"].to(device=device, dtype=dtype)
            k2 = dynamic_overlap_cn["k2"].to(device=device, dtype=dtype)
            k3 = dynamic_overlap_cn["k3"].to(device=device, dtype=dtype)
            r_cov_loc = dynamic_overlap_cn["r_cov"].to(device=device, dtype=dtype)
            k_cn_loc = float(dynamic_overlap_cn["k_cn"])
            q_scf = dynamic_overlap_cn["q_scf"].to(device=device, dtype=dtype)
            q_eeq = dynamic_overlap_cn["q_eeqbc"].to(device=device, dtype=dtype)
            from ..basis.qvszp import compute_effective_charge
            q_eff_local = compute_effective_charge(numbers, positions, q_scf, q_eeq, r_cov=r_cov_loc, k_cn=k_cn_loc, k0=k0, k1=k1, k2=k2, k3=k3)
            use_qeff_coeff = True
        except Exception:
            use_qeff_coeff = False
    # Iterate shell pairs
    dist = torch.cdist(positions, positions)
    for i, shi in enumerate(shells):
        A = shi.atom_index; ZA = shi.element; lA = shi.l
        oi, ni = ao_off[i], ao_cnt[i]
        epsA = shell_eps(i)
        kWA = eht.get(f'k_w_{lA}', eht.get('k_w_s', torch.ones_like(P)))[ZA]
        for j, shj in enumerate(shells):
            if j <= i:
                continue
            B = shj.atom_index; ZB = shj.element; lB = shj.l
            oj, nj = ao_off[j], ao_cnt[j]
            epsB = shell_eps(j)
            kWB = eht.get(f'k_w_{lB}', eht.get('k_w_s', torch.ones_like(P)))[ZB]
            kpair = 0.5 * (kWA + kWB) if wolfsberg_mode != 'geometric' else torch.sqrt(torch.clamp(kWA, min=0.0) * torch.clamp(kWB, min=0.0))
            # Prefactors independent of R for this term (Eq. 68 third term)
            R_AB = dist[A, B]
            Pi_R = distance_factor(lA, lB, ZA, ZB, R_AB, eht)
            Xpen = en_penalty(ZA, ZB, en, k_en)
            avg_eps = 0.5 * (epsA + epsB)
            pref = (kpair * Xpen * Pi_R * avg_eps)
            # If P block negligible, skip
            Pblk = P[oi:oi+ni, oj:oj+nj]
            if torch.all(Pblk.abs() < 1e-15):
                continue
            # Build coefficients
            ai = alpha[i]; aj = alpha[j]
            if use_qeff_coeff and q_eff_local is not None:
                ci = c0[i] + torch.tensor([p[2] for p in shi.primitives], dtype=dtype, device=device) * q_eff_local[A]
                cj = c0[j] + torch.tensor([p[2] for p in shj.primitives], dtype=dtype, device=device) * q_eff_local[B]
            else:
                ci = c0[i]; cj = c0[j]
            # Differentiable overlap and diatomic scaling contraction
            Rvec = (positions[A] - positions[B]).detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
            S_sph = _ov_sph_torch(lmap[lA], lmap[lB], ai, ci, aj, cj, Rvec)
            if diat is None:
                S_sc = S_sph
            else:
                kA = {"sigma": float(diat['sigma'][ZA].item()), "pi": float(diat['pi'][ZA].item()), "delta": float(diat['delta'][ZA].item())}
                kB = {"sigma": float(diat['sigma'][ZB].item()), "pi": float(diat['pi'][ZB].item()), "delta": float(diat['delta'][ZB].item())}
                S_sc = _scale_s(S_sph, Rvec, lA, lB, kA, kB)
            val = (Pblk.to(dtype=dtype, device=device) * S_sc).sum()
            gR, = torch.autograd.grad(val, Rvec, retain_graph=False, create_graph=False)
            # Map to atoms: ∂/∂R_A = ∂/∂R, ∂/∂R_B = -∂/∂R
            dE_s[A] = dE_s[A] + pref.to(dtype=dtype) * gR
            dE_s[B] = dE_s[B] - pref.to(dtype=dtype) * gR
            # Explicit reverse block contribution (oj,oi) with Rrev = -R
            Pblk_r = P[oj:oj+nj, oi:oi+ni]
            if torch.any(Pblk_r.abs() > 1e-15):
                Rrev = (-Rvec).detach().clone().requires_grad_(True)
                S_sph_r = _ov_sph_torch(lmap[lB], lmap[lA], aj, cj, ai, ci, Rrev)
                if diat is None:
                    S_sc_r = S_sph_r
                else:
                    kA_r = {"sigma": float(diat['sigma'][ZB].item()), "pi": float(diat['pi'][ZB].item()), "delta": float(diat['delta'][ZB].item())}
                    kB_r = {"sigma": float(diat['sigma'][ZA].item()), "pi": float(diat['pi'][ZA].item()), "delta": float(diat['delta'][ZA].item())}
                    S_sc_r = _scale_s(S_sph_r, Rrev, lB, lA, kA_r, kB_r)
                val_r = (Pblk_r.to(dtype=dtype, device=device) * S_sc_r).sum()
                gR_r, = torch.autograd.grad(val_r, Rrev, retain_graph=False, create_graph=False)
                # Since Rrev = R_B - R_A, chain rule: ∂/∂R_A = -∂/∂Rrev, ∂/∂R_B = +∂/∂Rrev
                dE_s[A] = dE_s[A] - pref.to(dtype=dtype) * gR_r
                dE_s[B] = dE_s[B] + pref.to(dtype=dtype) * gR_r
    # Sum components
    dE = dE_eps + dE_pi + dE_s + dE_coeff
    # Ensure per-atom gradient shape (nat,3). If an intermediate inadvertently
    # introduced an extra axis over X atoms (nat,nat,3), reduce over that axis.
    if dE.dim() == 3 and dE.shape[0] == dE.shape[1]:
        dE = dE.sum(dim=1)
    if return_components:
        comps = {"d_eps": dE_eps, "d_pi": dE_pi, "d_s": dE_s, "d_coeff": dE_coeff}
        return dE, comps
    return dE
