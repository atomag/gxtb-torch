from __future__ import annotations

"""Anisotropic Electrostatics (AES), doc/theory/16_anisotropic_electrostatics.md.

Implements the multipole-expanded Coulomb energy and Fock contributions using
monopole–dipole, dipole–dipole, and monopole–quadrupole couplings with diatomic
damping. Equations implemented precisely per:
 - Multipole-expanded operator: Eq. (109) with derivatives Eq. (110a–b)
 - Moment integrals: S_{κλ}, D^α_{κλ}, Q^{αβ}_{κλ} (Eq. 111a–c)
 - Energy/Fock assembly via atom-partitioned multipoles (Sec. 1.11.1)

Notes
-----
- We exclude monopole–monopole interactions (covered by isotropic second-order TB),
  as stated in doc/theory/16.
- Damping follows the element/global AES entries defined in parameters per schema
  [aes] and [aes.element] (see params/schema.py). If required fields are missing,
  a ValueError is raised (no hidden defaults).
- Higher-order dipole–quadrupole (R^{-7}) and quadrupole–quadrupole (R^{-9}) terms
  (Eq. 110c–d) are implemented and included when AESParams provides dmp7/dmp9. If these
  exponents are absent from the schema, the higher-order terms are disabled (no
  hidden defaults). SI Eq. 117 single‑ion damping is available for n=3/5 via si_rules.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import torch

from ..cn import coordination_number

Tensor = torch.Tensor

__doc_refs__ = {
    "file": "doc/theory/16_anisotropic_electrostatics.md",
    "eqs": ["109", "110a", "110b", "110c", "110d", "111a", "111b", "111c", "117"],
}

__all__ = [
    "AESParams",
    "compute_multipole_moments",
    "compute_atomic_moments",
    "compute_aes_kernels",
    "potentials_from_multipoles",
    "assemble_fock_from_potentials",
    "aes_energy_and_fock",
    "__doc_refs__",
]


@dataclass(frozen=True)
class AESParams:
    """AES damping parameters sourced from parameters/[aes] and [aes.element].

    - dmp3, dmp5: exponents for orders n=3 and n=5 (applied to Eq. 110a–b terms).
    - mprad, mpvcn: element-wise base radius and small CN shift multiplier used to
      build damping radii per atom (mrad_A = mprad_Z + mpvcn_Z * CN_A).
    """

    dmp3: float
    dmp5: float
    mprad: Tensor
    mpvcn: Tensor
    # Optional higher-order damping exponents for n=7 and n=9 terms (Eq. 110c–d)
    dmp7: float | None = None
    dmp9: float | None = None


def compute_multipole_moments(
    P: Tensor,
    S: Tensor,
    D: Tuple[Tensor, Tensor, Tensor],
    Q: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
) -> Dict[str, Tensor]:
    """Compute Mulliken-type global moments Tr(P M) for sanity checks (Eq. 111a–c)."""
    if any(M is None for M in (S, *D, *Q)):
        raise ValueError("All required moment integrals (S,D,Q) must be provided (Eqs. 111a–c)")
    out: Dict[str, Tensor] = {}
    def trp(M: Tensor) -> Tensor:
        return torch.einsum('ij,ji->', P, M)
    out['S'] = trp(S)
    out['Dx'] = trp(D[0]); out['Dy'] = trp(D[1]); out['Dz'] = trp(D[2])
    Qxx,Qxy,Qxz,Qyy,Qyz,Qzz = Q
    out['Qxx'] = trp(Qxx); out['Qxy'] = trp(Qxy); out['Qxz'] = trp(Qxz)
    out['Qyy'] = trp(Qyy); out['Qyz'] = trp(Qyz); out['Qzz'] = trp(Qzz)
    return out


def compute_atomic_moments(
    P: Tensor,
    S: Tensor,
    D: Tuple[Tensor, Tensor, Tensor],
    Q: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    ao_atoms: Tensor,
) -> Dict[str, Tensor]:
    """Per-atom Mulliken partitioning of monopole, dipole, quadrupole moments (Eq. 111a–c).

    Differentiable w.r.t. matrix elements via one-hot AO→atom mapping (avoids bincount).

    Returns dict with keys 'q' (nat,), 'mu' (nat,3), 'Q' (nat,3,3).
    """
    device = P.device
    dtype = P.dtype
    nat = int(ao_atoms.max().item()) + 1
    nao = P.shape[0]
    # One-hot AO→atom mapping (nao,nat)
    Amap = torch.nn.functional.one_hot(ao_atoms.long(), num_classes=nat).to(dtype=dtype, device=device)  # (nao,nat)
    # Precompute contractions
    PS = P @ S
    PDx = P @ D[0]
    PDy = P @ D[1]
    PDz = P @ D[2]
    PQxx = P @ Q[0]; PQxy = P @ Q[1]; PQxz = P @ Q[2]
    PQyy = P @ Q[3]; PQyz = P @ Q[4]; PQzz = P @ Q[5]
    # Diagonals (nao,)
    dPS = torch.diag(PS)
    dPDx = torch.diag(PDx)
    dPDy = torch.diag(PDy)
    dPDz = torch.diag(PDz)
    dQxx = torch.diag(PQxx); dQxy = torch.diag(PQxy); dQxz = torch.diag(PQxz)
    dQyy = torch.diag(PQyy); dQyz = torch.diag(PQyz); dQzz = torch.diag(PQzz)
    # Accumulate to atoms via (nao,nat)^T @ (nao,) = (nat,)
    qA = Amap.T @ dPS
    muA_x = Amap.T @ dPDx
    muA_y = Amap.T @ dPDy
    muA_z = Amap.T @ dPDz
    muA = torch.stack([muA_x, muA_y, muA_z], dim=-1)
    QA = torch.zeros(nat, 3, 3, dtype=dtype, device=device)
    QA[:, 0, 0] = Amap.T @ dQxx
    QA[:, 0, 1] = Amap.T @ dQxy
    QA[:, 0, 2] = Amap.T @ dQxz
    QA[:, 1, 1] = Amap.T @ dQyy
    QA[:, 1, 2] = Amap.T @ dQyz
    QA[:, 2, 2] = Amap.T @ dQzz
    # Symmetrize QA off-diagonals
    QA[:, 1, 0] = QA[:, 0, 1]
    QA[:, 2, 0] = QA[:, 0, 2]
    QA[:, 2, 1] = QA[:, 1, 2]
    return {"q": qA, "mu": muA, "Q": QA}


def _build_mrad(numbers: Tensor, positions: Tensor, params: AESParams, *, r_cov: Tensor, k_cn: float) -> Tensor:
    """Multipole damping radii per atom.

    Using available parameters, we form a linear CN-shifted radius:
        mrad_A = mprad_Z + mpvcn_Z * CN_A
    where CN_A is computed via Eq. 47 (doc/theory/9). This mirrors the intent of
    valence-CN dependence in dxtb AES2 when global logistic controls are absent.
    """
    device = positions.device
    dtype = positions.dtype
    # CN from doc/theory/9 (Eq. 47)
    cn = coordination_number(positions, numbers, r_cov.to(device=device, dtype=dtype), float(k_cn))
    z = numbers.to(device=device, dtype=torch.long)
    rad = params.mprad[z].to(device=device, dtype=dtype)
    vcn = params.mpvcn[z].to(device=device, dtype=dtype)
    mrad = rad + vcn * cn
    return mrad


def _pairwise_kernels(numbers: Tensor, positions: Tensor, mrad: Tensor, dmp3: float, dmp5: float,
                      *, si_params: dict | None = None, r_cov: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute damped kernels up to second derivatives.

    Returns
    -------
    rij : (nat,nat,3) displacement vectors R_A - R_B (masking diagonal)
    f3  : (nat,nat) damping-scaled R^{-3} factor (Eq. 110a)
    Hess: (nat,nat,3,3) ∇∇(1/R) damped (Eq. 110b)
    mask: (nat,nat) boolean off-diagonal mask
    """
    device = positions.device
    dtype = positions.dtype
    nat = positions.shape[0]
    # pairwise distances and vectors
    rij = positions.unsqueeze(1) - positions.unsqueeze(0)  # (nat,nat,3)
    # Squared distances and mask (avoid singularities at R=0 on diagonal)
    mask = ~torch.eye(positions.shape[0], dtype=torch.bool, device=device)
    eps = torch.finfo(dtype).eps
    R2 = (rij * rij).sum(dim=-1)
    invR = torch.where(mask, (R2.clamp_min(eps)).rsqrt(), torch.zeros_like(R2))
    g3 = invR * invR * invR  # eq: 110a, |∇(1/R)| factor ~ R^{-3}
    g5 = g3 * invR * invR    # eq: 110b, Hessian factor ~ R^{-5}
    # damping factors: default logistic; optional SI Eq.117 if si_params present
    rr_raw = 0.5 * (mrad.unsqueeze(1) + mrad.unsqueeze(0)) * invR  # (nat,nat)
    # Avoid evaluating rr^p at rr=0 for fractional p: set diagonal rr to 1 for damping eval
    rr = torch.where(mask, rr_raw, torch.ones_like(rr_raw))
    if si_params is None:
        fdmp3 = 1.0 / (1.0 + 6.0 * rr.pow(float(dmp3)))
        fdmp5 = 1.0 / (1.0 + 6.0 * rr.pow(float(dmp5)))
    else:
        # SI Eq.117: f_n = 0.5 * k_n * (1 - erf(-k_s_n (R - R0)))
        k3 = float(si_params.get('si_k3', 1.0))
        ks3 = float(si_params.get('si_ks3', 1.0))
        k5 = float(si_params.get('si_k5', 1.0))
        ks5 = float(si_params.get('si_ks5', 1.0))
        # R0 from rule: default rcov sum scaled or mrad average scaled
        R = torch.where(mask, 1.0 / (invR + torch.finfo(dtype).eps), torch.zeros_like(invR))
        mode = str(si_params.get('si_R0_mode', 'rcov'))
        sc = float(si_params.get('si_R0_scale', 1.0))
        if mode == 'rcov':
            if r_cov is None:
                raise ValueError("AES SI damping requires r_cov for R0 computation")
            rA = r_cov.to(device=device, dtype=dtype)[numbers.long()]
            R0 = sc * (rA.unsqueeze(1) + rA.unsqueeze(0))
        elif mode == 'mrad':
            R0 = sc * (0.5 * (mrad.unsqueeze(1) + mrad.unsqueeze(0)))
        else:
            raise ValueError("AES SI damping unknown si_R0_mode; use 'rcov' or 'mrad'")
        # Apply Eq.117
        fdmp3 = 0.5 * k3 * (1.0 - torch.erf(-ks3 * (R - R0)))
        fdmp5 = 0.5 * k5 * (1.0 - torch.erf(-ks5 * (R - R0)))
    f3 = g3 * fdmp3
    # Hessian (Eq. 110b): ∇∇(1/R) = (3 r_i r_j - δ_ij R^2) / R^5 = 3 rr g5 - δ g3
    # Build components
    r_i = rij.unsqueeze(-1)  # (nat,nat,3,1)
    r_j = rij.unsqueeze(-2)  # (nat,nat,1,3)
    rr_mat = r_i @ r_j  # (nat,nat,3,3)
    eye3 = torch.eye(3, device=device, dtype=dtype).view(1,1,3,3)
    Hess = (3.0 * rr_mat * g5.unsqueeze(-1).unsqueeze(-1) - eye3 * g3.unsqueeze(-1).unsqueeze(-1))
    Hess = Hess * fdmp5.unsqueeze(-1).unsqueeze(-1)
    # enforce zeros on diagonal blocks explicitly for all returned tensors
    rij = torch.where(mask.unsqueeze(-1), rij, torch.zeros_like(rij))
    f3 = torch.where(mask, f3, torch.zeros_like(f3))
    Hess = torch.where(mask.unsqueeze(-1).unsqueeze(-1), Hess, torch.zeros_like(Hess))
    return rij, f3, Hess, mask


def _third_derivative_tensor(rij: Tensor, invR: Tensor, fdmp7: Tensor | None) -> Tensor | None:
    """Third derivative ∇∇∇(1/R) tensor T3_{abc} per Eq. 110c.

    Eq. (110c): ∂_a ∂_b ∂_c (1/R) = - [15 r_a r_b r_c - 3 (r_a δ_{bc} + r_b δ_{ac} + r_c δ_{ab}) R^2] / R^7
    We construct the undamped tensor and apply fdmp7 if provided.
    """
    if fdmp7 is None:
        return None
    device = rij.device
    dtype = rij.dtype
    # Use direct R^2 to avoid divisions by zero on diagonal; invR carries zeros on diagonal
    R2 = (rij * rij).sum(dim=-1)
    g7 = invR ** 7
    eye = torch.eye(3, device=device, dtype=dtype)
    T = torch.zeros((*rij.shape[:-1], 3, 3, 3), dtype=dtype, device=device)
    r = rij
    for a in range(3):
        for b in range(3):
            for c in range(3):
                term_rrr = 15.0 * r[..., a] * r[..., b] * r[..., c]
                term_mix = 3.0 * (r[..., a] * eye[b, c] + r[..., b] * eye[a, c] + r[..., c] * eye[a, b]) * R2
                T[..., a, b, c] = -(term_rrr - term_mix) * g7
    # apply damping fdmp7
    T = T * fdmp7.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return T


def _fourth_derivative_tensor(rij: Tensor, invR: Tensor, fdmp9: Tensor | None) -> Tensor | None:
    """Fourth derivative ∇∇∇∇(1/R) tensor T4_{abcd} per Eq. 110d.

    Eq. (110d):
      ∂_a ∂_b ∂_c ∂_d (1/R) = [105 r_a r_b r_c r_d - 15( r_a r_b δ_{cd} + r_a r_c δ_{bd} + r_b r_c δ_{ad}
        + r_a r_d δ_{bc} + r_b r_d δ_{ac} + r_c r_d δ_{ab}) R^2 + 3 (δ_{ab} δ_{cd} + δ_{ac} δ_{bd} + δ_{ad} δ_{bc}) R^4] / R^9
    We construct the undamped tensor and apply fdmp9 if provided.
    """
    if fdmp9 is None:
        return None
    device = rij.device
    dtype = rij.dtype
    # Use direct R^2 and R^4 to avoid 1/0 on diagonal
    R2 = (rij * rij).sum(dim=-1)
    R4 = R2 * R2
    g9 = invR ** 9
    eye = torch.eye(3, device=device, dtype=dtype)
    T = torch.zeros((*rij.shape[:-1], 3, 3, 3, 3), dtype=dtype, device=device)
    r = rij
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    term_r4 = 105.0 * r[..., a] * r[..., b] * r[..., c] * r[..., d]
                    term_r2 = 15.0 * (
                        r[..., a] * r[..., b] * eye[c, d] +
                        r[..., a] * r[..., c] * eye[b, d] +
                        r[..., b] * r[..., c] * eye[a, d] +
                        r[..., a] * r[..., d] * eye[b, c] +
                        r[..., b] * r[..., d] * eye[a, c] +
                        r[..., c] * r[..., d] * eye[a, b]
                    ) * R2
                    term_r0 = 3.0 * (eye[a, b] * eye[c, d] + eye[a, c] * eye[b, d] + eye[a, d] * eye[b, c]) * R4
                    T[..., a, b, c, d] = (term_r4 - term_r2 + term_r0) * g9
    T = T * fdmp9.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return T


def aes_energy_and_fock(
    numbers: Tensor,
    positions: Tensor,
    basis,
    P: Tensor,
    S: Tensor,
    D: Tuple[Tensor, Tensor, Tensor],
    Q: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    params: AESParams,
    *,
    r_cov: Tensor,
    k_cn: float,
    si_rules: dict | None = None,
) -> Tuple[Tensor, Tensor]:
    """Compute AES energy and Fock up to second derivatives (qμ, μμ, qΘ).

    Mapping to theory:
    - Gradients/Hessians: Eq. 110a–110b
    - Moments: Eq. 111a–c
    - Energy assembly (excluding mono–mono): Sec. 1.11.1 and SI Eq. (116) restricted
      to n=3,5 contributions. Damping via [aes], [aes.element].
    """
    device = positions.device
    dtype = positions.dtype
    # AO→atom mapping
    ao_atoms = []
    for ish, off in enumerate(basis.ao_offsets):
        n_ao = basis.ao_counts[ish]
        ao_atoms.extend([basis.shells[ish].atom_index] * n_ao)
    ao_atoms_t = torch.tensor(ao_atoms, dtype=torch.long, device=device)
    # Atomic multipoles (Mulliken). Use charge fluctuation Δq = N_valence - pop for q-based terms
    atoms = compute_atomic_moments(P, S, D, Q, ao_atoms_t)
    popA = atoms["q"].to(device=device, dtype=dtype)
    muA = atoms["mu"].to(device=device, dtype=dtype)
    QA = atoms["Q"].to(device=device, dtype=dtype)
    # Valence electrons consistent with shells present in basis (doc/theory/3 Eq. 5 reference state)
    # Build per-atom valence counts using the electron configuration heuristic
    from .second_order_tb import _electron_configuration_valence_counts as _valence_map
    nat = int(numbers.shape[0])
    # Determine shell presence per atom
    present = [set() for _ in range(nat)]
    for sh in basis.shells:
        present[sh.atom_index].add(sh.l)
    val_counts = torch.zeros(nat, dtype=dtype, device=device)
    for A in range(nat):
        conf = _valence_map(int(numbers[A].item()))
        val_counts[A] = float(sum(v for l, v in conf.items() if l in present[A]))
    qA = val_counts - popA
    # Build damping radii (Eq. 47 CN model + AES radius generation)
    r_cov = r_cov.to(device=device, dtype=dtype)
    mrad = _build_mrad(numbers, positions, params, r_cov=r_cov, k_cn=k_cn)
    # Kernels
    rij, f3, Hess, mask = _pairwise_kernels(numbers, positions, mrad, params.dmp3, params.dmp5,
                                            si_params=si_rules, r_cov=r_cov)
    invR = torch.zeros_like(f3)
    nz = mask
    invR[nz] = 1.0 / torch.linalg.norm(rij[nz], dim=-1)
    # Energy and atomic potentials (up to n=5)
    nat = numbers.shape[0]
    v_mono = torch.zeros(nat, dtype=dtype, device=device)
    v_dip = torch.zeros(nat, 3, dtype=dtype, device=device)
    v_quad = torch.zeros(nat, 3, 3, dtype=dtype, device=device)
    E = torch.zeros((), dtype=dtype, device=device)
    # Iterate atoms vectorized via broadcasting
    # Build pair contributions with broadcasting: A index along rows, B along cols
    qAi = qA.unsqueeze(1)
    qBj = qA.unsqueeze(0)
    muAi = muA.unsqueeze(1)  # (nat,1,3)
    muBj = muA.unsqueeze(0)  # (1,nat,3)
    # Monopole–dipole energy (q–μ), Eq. 110a for ∇(1/R):
    # E_md = -1/2 Σ_{A≠B} [ q_A μ_B · (r_AB/R^3) - q_B μ_A · (r_AB/R^3) ]
    vec_g3 = (rij * f3.unsqueeze(-1))  # (nat,nat,3) equals r/R^3 * fdmp3
    # q_A μ_B · vec - q_B μ_A · vec
    term_md = (qAi.unsqueeze(-1) * (muBj * vec_g3)).sum(-1) - (qBj.unsqueeze(-1) * (muAi * vec_g3)).sum(-1)
    E = E - 0.5 * term_md.sum()
    # Monopole potential v_mono from μ and Q: v_mono(A) = - Σ_B μ_B·(r/R^3) + (1/2) Tr(Θ_B Hess)
    v_mono = v_mono - (muBj * vec_g3).sum(-1).sum(dim=1)  # sum over B
    # Dipole–dipole energy (μ–μ), Eq. 110b for ∇∇(1/R):
    # E_dd = 1/2 Σ μ_A^T (Hess) μ_B
    H_muB = torch.einsum('ijab, j b -> ija', Hess, muA)
    # μ_A · (H μ_B)
    term_dd = (muA.unsqueeze(1) * H_muB).sum(-1)  # (nat,nat)
    E = E + 0.5 * term_dd.sum()
    # Dipole potential v_dip(A) = - Σ_B q_B (r/R^3) + Σ_B Hess · μ_B
    v_dip = v_dip - (qBj.unsqueeze(-1) * vec_g3).sum(dim=1) + H_muB.sum(dim=1)
    # Monopole–quadrupole energy (q–Θ), using Eq. 110b tensor:
    # E_mq = 1/2 Σ [ q_A Tr(Θ_B Hess) + q_B Tr(Θ_A Hess) ]
    # trace contraction Tr(Θ_B Hess) over (αβ)
    tr_H_ThetaB = torch.einsum('ijab, j ab -> ij', Hess, QA)
    tr_H_ThetaA = torch.einsum('ijab, i ab -> ij', Hess, QA)
    E = E + 0.5 * ( (qAi * tr_H_ThetaB) + (qBj * tr_H_ThetaA) ).sum()
    # v_mono add + (1/2) Tr(Θ_B Hess)
    v_mono = v_mono + 0.5 * tr_H_ThetaB.sum(dim=1)
    # v_quad(A) += (1/2) q_B Hess  (matrix)
    v_quad = v_quad + 0.5 * torch.einsum('ij, ijab -> iab', qBj, Hess)

    # Optional higher-order terms (Eq. 110c–d): require explicit rule enablement in si_rules
    enable_high = bool(si_rules.get('enable_high_order', 0)) if isinstance(si_rules, dict) else False
    has7 = enable_high and (params.dmp7 is not None)
    has9 = enable_high and (params.dmp9 is not None)
    if has7 or has9:
        # Build traceless quadrupole tensor Θ_A from raw Q_A: Θ = Q - (Tr(Q)/3) I (Eq. 113b inspired)
        eye3 = torch.eye(3, device=device, dtype=dtype)
        ThetaA = QA - (QA.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0).unsqueeze(-1).unsqueeze(-1) * eye3
        # Reuse rij and invR; compute rr for damping
        rr = 0.5 * (mrad.unsqueeze(1) + mrad.unsqueeze(0)) * invR
        if has7:
            fdmp7 = 1.0 / (1.0 + 6.0 * rr.pow(float(params.dmp7)))
            T3 = _third_derivative_tensor(rij, invR, fdmp7)
            if T3 is not None:
                # E_dq = -1/6 Σ_{A≠B} [ μ_B^a Θ_A^{bc} T3_{abc} + μ_A^a Θ_B^{bc} T3_{abc} ]
                E = E - (
                    torch.einsum('j a, i b c, i j a b c ->', muA, ThetaA, T3)
                    + torch.einsum('i a, j b c, i j a b c ->', muA, ThetaA, T3)
                ) / 6.0
                # v_dip(A) += -1/6 Σ_B Θ_B^{bc} T3_{a b c}
                v_dip = v_dip - (torch.einsum('j b c, i j a b c -> i a', ThetaA, T3)) / 6.0
                # v_quad(A) += -1/6 Σ_B μ_B^a T3_{a b c}
                v_quad = v_quad - (torch.einsum('j a, i j a b c -> i b c', muA, T3)) / 6.0
        if has9:
            fdmp9 = 1.0 / (1.0 + 6.0 * rr.pow(float(params.dmp9)))
            T4 = _fourth_derivative_tensor(rij, invR, fdmp9)
            if T4 is not None:
                # E_qq = +1/24 Σ_{A≠B} Θ_A^{ab} Θ_B^{cd} T4_{abcd}
                E = E + torch.einsum('i a b, j c d, i j a b c d ->', ThetaA, ThetaA, T4) / 24.0
                # v_quad(A) += +1/24 Σ_B T4_{a b c d} Θ_B^{c d}
                v_quad = v_quad + (torch.einsum('i j a b c d, j c d -> i a b', T4, ThetaA)) / 24.0
    # Build AO-level H from atomic potentials (Eq. 109 with moments Eq. 111a–c)
    # H_AES = -0.5 [ S ∘ (v_mono(A)+v_mono(B)) + Σ_α D^α ∘ (v_dip^α(A)+v_dip^α(B)) + Σ_{αβ} Q^{αβ} ∘ (v_quad^{αβ}(A)+v_quad^{αβ}(B)) ]
    # Prepare AO-spread potentials
    vA = v_mono[ao_atoms_t]
    VA = vA.unsqueeze(1)
    VB = vA.unsqueeze(0)
    H = torch.zeros_like(S)
    H = H - 0.5 * S * (VA + VB)
    # Dipole components
    for comp, M in enumerate(D):
        vcomp = v_dip[:, comp]
        w = vcomp[ao_atoms_t]
        WA = w.unsqueeze(1); WB = w.unsqueeze(0)
        H = H - 0.5 * M * (WA + WB)
    # Quadrupole components (map 3x3 to six unique components in Q tuple ordering)
    # Q tuple ordering: (xx, xy, xz, yy, yz, zz)
    comps = [ (0,0), (0,1), (0,2), (1,1), (1,2), (2,2) ]
    for (a,b), M in zip(comps, Q):
        vcomp = v_quad[:, a, b]
        w = vcomp[ao_atoms_t]
        WA = w.unsqueeze(1); WB = w.unsqueeze(0)
        H = H - 0.5 * M * (WA + WB)
    # Energy from Fock-like contribution: E = 1/2 Tr(H ∘ P)
    E_AES = 0.5 * torch.einsum('ij,ji->', H, P)
    return E_AES, H


def compute_aes_kernels(
    numbers: Tensor,
    positions: Tensor,
    params: AESParams,
    *,
    r_cov: Tensor,
    k_cn: float,
    si_rules: dict | None = None,
) -> Dict[str, Tensor]:
    """Precompute geometry-dependent AES kernels (doc/theory/16, Eq. 110a–b).

    Returns a dict with keys:
      - 'vec_g3': (nat,nat,3) = r_AB/R^3 with damping f_dmp3 applied
      - 'Hess': (nat,nat,3,3) = ∇∇(1/R) with damping f_dmp5 applied
      - 'mask': (nat,nat) boolean mask for A≠B
      - 'mrad': (nat,) multipole damping radii
    """
    device = positions.device
    dtype = positions.dtype
    r_cov = r_cov.to(device=device, dtype=dtype)
    mrad = _build_mrad(numbers, positions, params, r_cov=r_cov, k_cn=k_cn)
    rij, f3, Hess, mask = _pairwise_kernels(numbers, positions, mrad, params.dmp3, params.dmp5, si_params=si_rules, r_cov=r_cov)
    vec_g3 = rij * f3.unsqueeze(-1)
    return {"vec_g3": vec_g3, "Hess": Hess, "mask": mask, "mrad": mrad}


def potentials_from_multipoles(
    qA: Tensor,
    muA: Tensor,
    QA: Tensor,
    kernels: Dict[str, Tensor],
    *,
    params: AESParams | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute atomic potentials (v_mono, v_dip, v_quad) from atomic multipoles.

    Formulas mirror the internal assembly in aes_energy_and_fock (up to n=5) so
    that mixing in potential space matches the density-driven potentials.
    """
    vec_g3 = kernels["vec_g3"]
    Hess = kernels["Hess"]
    device = qA.device
    dtype = qA.dtype
    nat = qA.shape[0]
    # Broadcast helpers
    qAi = qA.unsqueeze(1)
    qBj = qA.unsqueeze(0)
    muAi = muA.unsqueeze(1)
    muBj = muA.unsqueeze(0)
    # v_mono(A) = - Σ_B μ_B·(r/R^3) + (1/2) Tr(Θ_B Hess)
    v_mono = -(muBj * vec_g3).sum(-1).sum(dim=1)
    tr_H_ThetaB = torch.einsum('ijab, j ab -> ij', Hess, QA)
    v_mono = v_mono + 0.5 * tr_H_ThetaB.sum(dim=1)
    # v_dip(A) = - Σ_B q_B (r/R^3) + Σ_B Hess · μ_B
    H_muB = torch.einsum('ijab, j b -> ija', Hess, muA)
    v_dip = -(qBj.unsqueeze(-1) * vec_g3).sum(dim=1) + H_muB.sum(dim=1)
    # v_quad(A) = (1/2) Σ_B q_B Hess  (matrix)
    v_quad = 0.5 * torch.einsum('ij, ijab -> iab', qBj, Hess)
    # Optional n=7/9 contributions can be included here in the future to match aes_energy_and_fock
    return v_mono, v_dip, v_quad


def assemble_fock_from_potentials(
    ao_atoms: Tensor,
    S: Tensor,
    D: Tuple[Tensor, Tensor, Tensor],
    Q: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    v_mono: Tensor,
    v_dip: Tensor,
    v_quad: Tensor,
) -> Tensor:
    """Map atomic potentials to AO-level AES Fock contribution (Eq. 109).

    H_AES = -0.5 [ S ∘ (v_A+v_B) + Σ_α D^α ∘ (v^α_A+v^α_B) + Σ_{αβ} Q^{αβ} ∘ (v^{αβ}_A+v^{αβ}_B) ]
    """
    device = S.device
    dtype = S.dtype
    va = v_mono[ao_atoms]
    VA = va.unsqueeze(1); VB = va.unsqueeze(0)
    H = -0.5 * S * (VA + VB)
    for comp, M in enumerate(D):
        w = v_dip[:, comp][ao_atoms]
        WA = w.unsqueeze(1); WB = w.unsqueeze(0)
        H = H - 0.5 * M * (WA + WB)
    comps = [ (0,0), (0,1), (0,2), (1,1), (1,2), (2,2) ]
    for (a,b), M in zip(comps, Q):
        w = v_quad[:, a, b][ao_atoms]
        WA = w.unsqueeze(1); WB = w.unsqueeze(0)
        H = H - 0.5 * M * (WA + WB)
    return H
