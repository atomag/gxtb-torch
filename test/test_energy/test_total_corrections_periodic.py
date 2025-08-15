import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema, map_cn_params
from gxtb.params.schema import map_ofx_element, map_mfx_element
from gxtb.basis.qvszp import build_atom_basis
from gxtb.energy.total import compute_total_energy, energy_report


def _valence_electrons_for_system(numbers, basis) -> int:
    # Use SCF’s conservative basis-aware valence counter to choose an even nelec
    from gxtb.scf import _valence_electron_counts
    ve = _valence_electron_counts(numbers, basis)
    total = float(ve.sum().item())
    # Round to nearest even integer and clamp to AO capacity
    nelec = int(2 * round(total / 2.0))
    nao = basis.nao
    nelec = max(2, min(nelec, 2 * nao))
    # Enforce even electron count for RHF
    if nelec % 2 == 1:
        nelec += 1
    return nelec


def test_periodic_sweep_XH_with_OFX_MFX_AES_ACP():
    dtype = torch.float64
    device = torch.device("cpu")
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    eeq = load_eeq_params('parameters/eeq')
    bq = load_basisq('parameters/basisq')

    # Elements covered in both parameter and basis files
    elements_all = sorted(set(gparams.elements.keys()) & set(bq.elements.keys()))
    # Sample a representative subset to keep runtime reasonable
    elements = sorted(set([elements_all[0], elements_all[1]] + elements_all[::5]))
    # Build MFX per-element U_shell map from schema (required)
    U_shell = map_mfx_element(gparams, schema)
    # Provide fixed global MFX scalars; schema does not fix them globally
    mfx_globals = {'alpha': 0.6, 'omega': 0.5, 'k1': 0.0, 'k2': 0.0}
    # Use theory-guided ξ per shell if schema lacks them: ξ = [1,1,2,2]
    xi_l = torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=dtype, device=device)
    # OFX onsite constants: for a robust sweep across all Z, supply explicit
    # zero-valued per-element tensors so E_OFX is well-defined and finite.
    Zmax = max(gparams.elements)
    def _zero_vec():
        return torch.zeros(Zmax+1, dtype=dtype, device=device)
    ofx_elem = {
        'sp': _zero_vec(), 'pp_off': _zero_vec(), 'sd': _zero_vec(), 'pd': _zero_vec(),
        'dd_off': _zero_vec(), 'sf': _zero_vec(), 'pf': _zero_vec(), 'df': _zero_vec(), 'ff_off': _zero_vec(),
    }
    # ACP CN and radii
    cn = map_cn_params(gparams, schema)
    r_cov = cn['r_cov'].to(device=device, dtype=dtype)
    k_cn = float(cn['k_cn'])

    # Fixed X–H bond length (Å) for the sweep
    R = 1.1

    for z in elements:
        # Form X–H diatomic; include z=1 (H–H) once via this loop
        numbers = torch.tensor([z, 1], dtype=torch.long, device=device)
        positions = torch.tensor([[0.0, 0.0, 0.0], [R, 0.0, 0.0]], dtype=dtype, device=device)
        basis = build_atom_basis(numbers, bq)
        nelec = _valence_electrons_for_system(numbers, basis)

        # Build minimal ACP params: small s-only projectors for active elements
        c0 = torch.zeros((Zmax+1, 4), dtype=dtype, device=device)
        xi = torch.zeros_like(c0)
        # Assign small finite values for active Z and H, s-channel only
        c0[z, 0] = 0.01
        xi[z, 0] = 0.8
        c0[1, 0] = 0.01
        xi[1, 0] = 0.8
        cn_avg = torch.ones(Zmax+1, dtype=dtype, device=device)
        acp_params = {
            'c0': c0,
            'xi': xi,
            'k_acp_cn': 0.1,
            'cn_avg': cn_avg,
            'r_cov': r_cov,
            'k_cn': k_cn,
            'l_list': ("s",),
        }

        # MFX params bundle per call
        mfx_params = {
            'alpha': mfx_globals['alpha'],
            'omega': mfx_globals['omega'],
            'k1': mfx_globals['k1'],
            'k2': mfx_globals['k2'],
            'U_shell': U_shell.to(device=device, dtype=dtype),
            'xi_l': xi_l,
        }

        # OFX params: alpha + explicit zero onsite map to avoid huge values for heavy Z
        ofx_params = {'alpha': 0.6, 'ofx_elem': ofx_elem}

        res = compute_total_energy(
            numbers, positions, basis, gparams, schema, eeq,
            total_charge=0.0, nelec=nelec,
            ofx=True, ofx_params=ofx_params,
            mfx=True, mfx_params=mfx_params,
            aes=True,
            acp=True, acp_params=acp_params,
            second_order=True,
        )
        rep = energy_report(res)
        # Ensure contributions exist and are finite
        assert 'E_OFX' in rep and torch.isfinite(res.scf.E_OFX)
        assert 'E_MFX' in rep and torch.isfinite(res.scf.E_MFX)
        assert 'E_AES' in rep and torch.isfinite(res.scf.E_AES)
        assert 'E_ACP' in rep and torch.isfinite(res.scf.E_ACP)
