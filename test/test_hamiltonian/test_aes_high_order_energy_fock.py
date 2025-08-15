import torch

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_aes_global, map_aes_element, map_cn_params
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.aes import AESParams, aes_energy_and_fock


def test_aes_high_order_energy_fock_consistency_h2():
    # Build a simple H2 system
    dtype = torch.float64
    device = torch.device("cpu")
    numbers = torch.tensor([1, 1], dtype=torch.int64, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=dtype, device=device)

    # Load parameters and schema
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    aesg = map_aes_global(gparams, schema)
    aese = map_aes_element(gparams, schema)
    cn = map_cn_params(gparams, schema)

    # Basis and AO moments
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    S, D, Q = build_moment_matrices(numbers, positions, basis)

    # Construct a symmetric density matrix P (arbitrary but consistent)
    nao = S.shape[0]
    # Simple choice: P = orthonormal projector onto first n/2 AOs (not SCF, but valid for test)
    evals, evecs = torch.linalg.eigh(S)
    X = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
    nocc = max(1, nao // 2)
    C = X  # assume orthonormal MOs ~ AOs
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T

    # AES params: base n=3,5 only
    params_lo = AESParams(dmp3=float(aesg['dmp3']), dmp5=float(aesg['dmp5']), mprad=aese['mprad'], mpvcn=aese['mpvcn'])
    # AES params: include derived n=7,9 if available from rules
    params_hi = AESParams(
        dmp3=float(aesg['dmp3']), dmp5=float(aesg['dmp5']), mprad=aese['mprad'], mpvcn=aese['mpvcn'],
        dmp7=float(aesg['dmp7']) if 'dmp7' in aesg else None,
        dmp9=float(aesg['dmp9']) if 'dmp9' in aesg else None,
    )

    # Compute AES E/H (low-order) and (high-order)
    E_lo, H_lo = aes_energy_and_fock(numbers, positions, basis, P, S, D, Q, params_lo, r_cov=cn['r_cov'], k_cn=cn['k_cn'])
    E_hi, H_hi = aes_energy_and_fock(numbers, positions, basis, P, S, D, Q, params_hi, r_cov=cn['r_cov'], k_cn=cn['k_cn'])

    # If derived dmp7/dmp9 present, Δ must satisfy ΔE ≈ 0.5 Tr(ΔH P)
    if 'dmp7' in aesg or 'dmp9' in aesg:
        dE = (E_hi - E_lo).item()
        dH = H_hi - H_lo
        rhs = 0.5 * torch.einsum('ij,ji->', dH, P).item()
        assert abs(dE - rhs) < 1e-8
