import torch

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_aes_global, map_aes_element, map_cn_params
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.aes import AESParams, aes_energy_and_fock


def test_aes_dmp79_explicit_indices_change_energy_vs_disabled():
    # CH system: compute AES with and without high-order terms using explicit schema indices
    dtype = torch.float64
    device = torch.device("cpu")
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    numbers = torch.tensor([6, 1], dtype=torch.long, device=device)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=dtype, device=device)
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    S, D, Q = build_moment_matrices(numbers, positions, basis)
    # simple symmetric P
    nao = S.shape[0]
    P = torch.eye(nao, dtype=dtype, device=device)
    # Map AES params
    aesg = map_aes_global(gparams, schema)
    aese = map_aes_element(gparams, schema)
    cn = map_cn_params(gparams, schema)
    # Construct params without high-order
    params_lo = AESParams(dmp3=float(aesg['dmp3']), dmp5=float(aesg['dmp5']), mprad=aese['mprad'], mpvcn=aese['mpvcn'])
    E_lo, _ = aes_energy_and_fock(numbers, positions, basis, P, S, D, Q, params_lo, r_cov=cn['r_cov'], k_cn=cn['k_cn'])
    # Construct params with explicit high-order indices
    assert 'dmp7' in aesg and 'dmp9' in aesg
    params_hi = AESParams(
        dmp3=float(aesg['dmp3']), dmp5=float(aesg['dmp5']), mprad=aese['mprad'], mpvcn=aese['mpvcn'],
        dmp7=float(aesg['dmp7']), dmp9=float(aesg['dmp9'])
    )
    E_hi, _ = aes_energy_and_fock(numbers, positions, basis, P, S, D, Q, params_hi, r_cov=cn['r_cov'], k_cn=cn['k_cn'])
    # With high-order enabled, energy should change deterministically
    assert abs((E_hi - E_lo).item()) > 0.0

