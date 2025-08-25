from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

import torch
import logging

try:
    from ase.calculators.calculator import Calculator, all_changes
except Exception:  # pragma: no cover
    Calculator = object  # type: ignore
    def all_changes():  # type: ignore
        return {"positions", "numbers", "cell", "pbc"}

from .device import get_device, set_default_dtype
from .params.loader import load_basisq, load_d4_parameters, load_eeq_params, load_gxtb_params
from .params.schema import (
    GxTBSchema,
    load_schema,
    map_cn_params,
    map_increment_params,
    map_repulsion_params,
    map_hubbard_params,
    map_qvszp_prefactors,
)
from .classical.increment import energy_increment
from .classical.repulsion import RepulsionParams, repulsion_energy
from .cn import coordination_number
from .basis.qvszp import build_atom_basis, compute_effective_charge_pbc, build_dynamic_primitive_coeffs
from .hamiltonian.scf_adapter import build_eht_core, make_core_builder
from .scf import scf
import gxtb.scf as _scf_mod
from .hamiltonian.second_order_tb import _electron_configuration_valence_counts
from .scf import _valence_electron_counts
from .io.molden import write as write_molden, shells_from_qvszp, MOSet, MOWavefunction
from .basis.qvszp import compute_effective_charge

EH2EV = 27.211386245988
logger = logging.getLogger(__name__)


class GxTBCalculator(Calculator):
    implemented_properties = ["energy"]  # extend to forces, stress later

    def __init__(
        self,
        parameters_dir: str = "parameters",
        schema_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float64,
        label: Optional[str] = None,
        enable_second_order: bool = True,
        enable_third_order: bool = False,
        enable_fourth_order: bool = False,
        enable_dynamic_overlap: bool = True,
        enable_aes: bool = False,
        # Dispersion (DFT-D4) controls
        enable_dispersion: bool = False,
        d4_variant: str = "bj-eeq-atm",
        d4_functional: Optional[str] = None,
        d4_reference_path: Optional[str] = None,
        d4_ref_dict: Optional[dict] = None,
        # Force evaluation controls (numerical differentiation)
        force_diff: str = "central",  # 'central' (2*3N evals) or 'forward' (1+3N evals)
        force_mode: str = "numeric",   # 'numeric' or 'analytic' (analytic supported for non-PBC)
        force_eps: float = 1.0e-3,
        force_log: bool = False,
        # PBC controls (see doc/theory/25_periodic_boundary_conditions.md)
        pbc_mode: Optional[str] = None,  # None, 'eht-gamma', 'eht-k', 'scf-k'
        gamma: Optional[bool] = None,    # convenience alias for pbc_mode='eht-gamma'
        kpoints: Optional[list] = None,  # list of fractional k-points [[kx,ky,kz], ...]
        kpoint_weights: Optional[list] = None,  # same length as kpoints; sum to 1.0
        mp_grid: Optional[tuple] = None,  # (n1,n2,n3) Monkhorst–Pack grid sizes (explicit)
        mp_shift: Optional[tuple] = None,  # (s1,s2,s3) shifts (explicit; e.g., 0 or 0.5)
        pbc_cutoff: Optional[float] = None,  # real-space cutoff (Angstrom) for S/H blocks
        pbc_cn_cutoff: Optional[float] = None,  # cutoff for CN under PBC
        pbc_disp_cutoff: Optional[float] = None,  # cutoff for dispersion under PBC
        pbc_aes_cutoff: Optional[float] = None,  # cutoff for AES lattice sums under PBC
        pbc_aes_high_order_cutoff: Optional[float] = None,  # optional cutoff for AES n=7/9 (falls back to pbc_aes_cutoff)
        # Ewald parameters for PBC second-order (required for scf-k)
        ewald_eta: Optional[float] = None,
        ewald_r_cut: Optional[float] = None,
        ewald_g_cut: Optional[float] = None,
        # SCF knobs (mixing and convergence)
        scf_mix: float = 0.5,
        scf_tol: float = 1e-6,
        scf_max_iter: int = 50,
        # Mixing controls (exposed): scheme and damping parameters passed to scf()
        scf_mixing: str = "anderson",           # 'linear' | 'anderson' | 'broyden'
        scf_mixing_history: int = 5,             # history length for Anderson/Broyden
        scf_beta_min: float = 0.05,              # lower bound for adaptive beta
        scf_beta_max: float = 0.8,               # upper bound for adaptive beta
        scf_beta_decay: float = 0.5,             # decay factor when divergence detected
        scf_restart_on_nan: bool = True,         # restart mixing with reduced beta on NaN energy
        scf_mix_init: float = 0.1,               # initial linear mixing for soft-start
        scf_anderson_soft_start: bool = True,     # use simple mixing for first `history` steps
        scf_anderson_diag_offset: float = 0.01,   # diagonal offset (ridge) for Anderson normal eqs
        scf_scp_mode: str = "charge",            # 'charge' or 'potential' (AES)
        # Overlap SPD control for k-space orthogonalization
        s_spd_floor: float = 1e-8,
        s_psd_adapt: bool = True,
        pbc_cutoff_max: Optional[float] = None,
        pbc_cutoff_step: float = 1.0,
        # Overlap SPD strictness (if True, error if S(k) remains indefinite after adaptation)
        s_strict_spd: bool = False,
        # PBC AES coupling: if False, add AES as post-SCF energy only (no SCF feedback)
        pbc_aes_couple: bool = False,
        # Molden output (non-PBC only)
        molden_path: Optional[str] = None,
        molden_spherical: bool = True,
        molden_norm: Optional[str] = None,  # None|'sqrt_sii'|'inv_sqrt_sii'
        # Spin/UHF controls for SCF and Molden output
        uhf: bool = False,
        nelec_alpha: Optional[int] = None,
        nelec_beta: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if hasattr(super(), "__init__"):
            super().__init__(label=label, **kwargs)  # type: ignore
        set_default_dtype(dtype)
        self.device = get_device(device)
        # Load parameter files strictly
        self._p_gxtb = load_gxtb_params(f"{parameters_dir}/gxtb")
        self._p_eeq = load_eeq_params(f"{parameters_dir}/eeq")
        self._p_basis = load_basisq(f"{parameters_dir}/basisq")
        self._p_d4 = load_d4_parameters(f"{parameters_dir}/dftd4parameters.toml")

        # Load schema that maps gxtb lines to semantic fields
        schema_file = Path(schema_path) if schema_path is not None else Path(parameters_dir) / "gxtb.schema.toml"
        if not schema_file.exists():
            raise FileNotFoundError(
                f"Missing parameter schema: {schema_file}. Please provide mapping indices for repulsion, CN, and increment."
            )
        self._schema: GxTBSchema = load_schema(schema_file)
        self._map_rep: Dict[str, Any] = map_repulsion_params(self._p_gxtb, self._schema)
        self._map_cn: Dict[str, Any] = map_cn_params(self._p_gxtb, self._schema)
        self._deinc: torch.Tensor = map_increment_params(self._p_gxtb, self._schema)
        # feature toggles
        self._enable_second_order = bool(enable_second_order)
        self._enable_third_order = bool(enable_third_order)
        self._enable_fourth_order = bool(enable_fourth_order)
        self._enable_dynamic_overlap = bool(enable_dynamic_overlap)
        self._enable_aes = bool(enable_aes)
        # Dispersion switches (no hidden defaults)
        self._enable_dispersion = bool(enable_dispersion)
        self._d4_variant = str(d4_variant)
        self._d4_functional = d4_functional
        self._d4_reference_path = d4_reference_path
        self._d4_ref_static = d4_ref_dict  # geometry-independent tensors; we'll attach 'cn' per geometry
        # PBC configuration
        # API convenience: gamma=True implies pbc_mode='eht-gamma' (explicit, not a hidden default)
        if gamma:
            if pbc_mode is None:
                pbc_mode = 'eht-gamma'
            elif pbc_mode != 'eht-gamma':
                raise ValueError("gamma=True conflicts with pbc_mode!=\"eht-gamma\"")
        self._pbc_mode = pbc_mode
        self._kpoints = kpoints
        self._kpoint_weights = kpoint_weights
        self._mp_grid = mp_grid
        self._mp_shift = mp_shift
        self._pbc_cutoff = pbc_cutoff
        self._pbc_cn_cutoff = pbc_cn_cutoff
        self._pbc_disp_cutoff = pbc_disp_cutoff
        self._pbc_aes_cutoff = pbc_aes_cutoff
        self._pbc_aes_high_order_cutoff = pbc_aes_high_order_cutoff
        self._ewald_eta = ewald_eta
        self._ewald_r_cut = ewald_r_cut
        self._ewald_g_cut = ewald_g_cut
        self._scf_mix = float(scf_mix)
        self._scf_tol = float(scf_tol)
        self._scf_max_iter = int(scf_max_iter)
        # Exposed mixing controls
        msch = str(scf_mixing).lower()
        if msch not in ("linear", "anderson", "broyden"):
            raise ValueError("scf_mixing must be 'linear', 'anderson', or 'broyden'")
        self._scf_mixing = msch
        self._scf_mixing_history = int(scf_mixing_history)
        self._scf_beta_min = float(scf_beta_min)
        self._scf_beta_max = float(scf_beta_max)
        self._scf_beta_decay = float(scf_beta_decay)
        self._scf_restart_on_nan = bool(scf_restart_on_nan)
        self._scf_mix_init = float(scf_mix_init)
        self._scf_anderson_soft_start = bool(scf_anderson_soft_start)
        self._scf_anderson_diag_offset = float(scf_anderson_diag_offset)
        scpm = str(scf_scp_mode).lower()
        if scpm not in ("charge", "potential"):
            raise ValueError("scf_scp_mode must be 'charge' or 'potential'")
        self._scf_scp_mode = scpm
        self._s_spd_floor = float(s_spd_floor)
        self._s_psd_adapt = bool(s_psd_adapt)
        self._pbc_cutoff_max = pbc_cutoff_max
        self._pbc_cutoff_step = float(pbc_cutoff_step)
        self._s_strict_spd = bool(s_strict_spd)
        self._pbc_aes_couple = bool(pbc_aes_couple)
        # Molden writer options
        self._molden_path = molden_path
        self._molden_spherical = bool(molden_spherical)
        self._molden_norm = molden_norm if molden_norm in (None, 'sqrt_sii', 'inv_sqrt_sii') else None
        # Force knobs
        self._force_diff = 'forward' if str(force_diff).lower().startswith('forw') else 'central'
        fm = str(force_mode).lower()
        if fm not in ("numeric", "analytic"):
            raise ValueError("force_mode must be 'numeric' or 'analytic'")
        self._force_mode = fm
        self._force_eps = float(force_eps)
        self._force_log = bool(force_log)
        # Spin options
        self._uhf = bool(uhf)
        self._nelec_alpha = nelec_alpha
        self._nelec_beta = nelec_beta
        # Warm-start state for numerical forces
        self._last_scf_q: Optional[torch.Tensor] = None
        self._last_scf_numbers: Optional[torch.Tensor] = None
        self._force_eval_mode: bool = False
        # SCF call counter for logging cycles
        self._scf_call_counter: int = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):  # type: ignore
        if hasattr(super(), "calculate"):
            super().calculate(atoms, properties, system_changes)  # type: ignore
        assert atoms is not None
        positions = torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype(), device=self.device)
        numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int64, device=self.device)

        # PBC guardrail and configuration validation (doc/theory/25_periodic_boundary_conditions.md)
        try:
            pbc_flags = atoms.get_pbc()
        except Exception:
            pbc_flags = (False, False, False)
        if any(bool(x) for x in tuple(pbc_flags)):
            # If Molden requested under PBC, raise (k-space wavefunctions not representable in Molden)
            if self._molden_path:
                raise NotImplementedError("Molden output is only supported for non-PBC calculations.")
            # Ensure explicit configuration was provided; no hidden defaults allowed.
            if self._pbc_mode is None:
                raise NotImplementedError(
                    "PBC requested by ASE Atoms.pbc, but no PBC mode configured. "
                    "Pass pbc_mode='eht-k' with explicit kpoints/kpoint_weights and pbc_cutoff as per doc/theory/25_periodic_boundary_conditions.md."
                )
            # Allow EHT band energies and SCF-k (with Ewald)
            if self._pbc_mode not in ("eht-gamma", "eht-k", "scf-k"):
                raise NotImplementedError(
                    f"Unsupported pbc_mode={self._pbc_mode!r}. Supported: 'eht-gamma' (Γ-only), 'eht-k' (band), 'scf-k' (k-SCF)."
                )
            # Required real-space cutoffs
            if self._pbc_cutoff is None or self._pbc_cutoff <= 0:
                raise ValueError("pbc_cutoff (real-space cutoff in Angstrom for S/H blocks) must be > 0.")
            if self._pbc_cn_cutoff is None or self._pbc_cn_cutoff <= 0:
                raise ValueError("pbc_cn_cutoff (cutoff for CN under PBC) must be > 0.")
            # Disallow unsupported terms explicitly (no hidden defaults)
            # AES allowed only for scf-k (requires density); block for EHT-only modes
            if self._enable_aes and self._pbc_mode != 'scf-k':
                raise NotImplementedError(
                    "AES under PBC requires k-SCF (density). Use pbc_mode='scf-k' and provide pbc_aes_cutoff."
                )
            # Periodic second order (Ewald-based) is only available in scf-k mode.
            # eq: doc/theory/25_periodic_boundary_conditions.md, doc/theory/15_second_order_tb.md (Ewald γ^(2)).
            if self._enable_second_order and self._pbc_mode != 'scf-k':
                raise NotImplementedError(
                    "Periodic second-order requires pbc_mode='scf-k' with explicit Ewald parameters (ewald_eta, ewald_r_cut, ewald_g_cut)."
                )

            # Assemble Bloch-sum EHT band energy (no SCF; doc/theory/25)
            from .pbc.cell import validate_cell
            from .pbc.kpoints import validate_kpoints, monkhorst_pack
            from .pbc.bloch import eht_lattice_blocks, assemble_k_matrices
            cell = validate_cell(torch.tensor(getattr(atoms, 'cell').array), pbc_flags)
            # Periodic CN (doc/theory/9 + 25) for use in onsite and optional dispersion
            from .pbc.cn_pbc import coordination_number_pbc as _cn_pbc
            cn = _cn_pbc(
                positions,
                numbers,
                self._map_cn["r_cov"].to(device=self.device, dtype=positions.dtype),
                float(self._map_cn["k_cn"]),
                cell,
                float(self._pbc_cn_cutoff),
            )
            # EEQBC charges required for dispersion (and later repulsion/second order under PBC)
            info = getattr(atoms, "info", {}) or {}
            q = info.get("q_eeqbc", None)
            if q is None and self._enable_dispersion:
                # Require explicit charges for PBC dispersion to avoid hidden defaults
                raise ValueError("PBC dispersion enabled but atoms.info['q_eeqbc'] missing. Provide EEQBC charges explicitly.")
            charges = torch.tensor(q if q is not None else [0.0] * int(numbers.shape[0]), dtype=positions.dtype, device=self.device)
            if self._pbc_mode in ("eht-k", "scf-k"):
                # Generate K from either explicit list or MP grid (both explicit; not both)
                have_list = isinstance(self._kpoints, list) and isinstance(self._kpoint_weights, list)
                have_mp = self._mp_grid is not None and self._mp_shift is not None
                if have_list and have_mp:
                    raise ValueError("Provide either (kpoints,kpoint_weights) or (mp_grid,mp_shift), not both.")
                if have_list:
                    K, W = validate_kpoints(self._kpoints, self._kpoint_weights)
                elif have_mp:
                    n1, n2, n3 = self._mp_grid  # type: ignore[misc]
                    s1, s2, s3 = self._mp_shift  # type: ignore[misc]
                    K, W = monkhorst_pack(int(n1), int(n2), int(n3), float(s1), float(s2), float(s3))
                else:
                    raise ValueError("For pbc_mode='eht-k', provide either (kpoints,kpoint_weights) or (mp_grid,mp_shift) explicitly.")
            else:  # 'eht-gamma'
                K = torch.zeros((1, 3), dtype=positions.dtype, device=self.device)
                W = torch.ones((1,), dtype=positions.dtype, device=self.device)
            # Basis setup
            basis = build_atom_basis(numbers, self._p_basis)
            # Build lattice blocks for EHT up to cutoffs
            # Possibly adapt pbc_cutoff until S(k) passes SPD threshold
            current_cut = float(self._pbc_cutoff)
            max_cut = float(self._pbc_cutoff_max or current_cut)
            step = float(self._pbc_cutoff_step)
            while True:
                blocks = eht_lattice_blocks(
                    numbers,
                    positions,
                    basis,
                    self._p_gxtb,
                    self._schema,
                    cell,
                    current_cut,
                    float(self._pbc_cn_cutoff),
                )
                mats_raw = assemble_k_matrices(blocks['translations'], blocks['S_blocks_raw'], blocks['H_blocks'], K)
                Sks_raw = mats_raw['S_k']; Hks = mats_raw['H_k']
                # SPD check on raw S(k)
                min_eig = min(float(torch.linalg.eigvalsh(Sk).real.min().item()) for Sk in Sks_raw)
                logger.debug("PBC S(k) min eigenvalue at cutoff %.3f Å: %.3e", current_cut, min_eig)
                if not self._s_psd_adapt or min_eig >= self._s_spd_floor or current_cut >= max_cut:
                    break
                current_cut = min(max_cut, current_cut + step)
                logger.debug("Adapting pbc_cutoff to %.3f Å due to SPD floor %.1e", current_cut, self._s_spd_floor)
            # Strict SPD enforcement (optional): raise if S(k) remains indefinite
            if self._s_strict_spd:
                min_eig_final = min(float(torch.linalg.eigvalsh(Sk).real.min().item()) for Sk in Sks_raw)
                if min_eig_final < self._s_spd_floor:
                    raise ValueError(
                        f"S(k) not SPD after cutoff adaptation (min eig={min_eig_final:.3e} < floor={self._s_spd_floor:.1e}). "
                        f"Increase pbc_cutoff_max or s_spd_floor to stabilize the overlap (doc/theory/25)."
                    )
            # Electron count heuristic consistent with non-periodic path (valence electrons present in basis)
            elem_to_shells: Dict[int, set[str]] = {}
            for sh in basis.shells:
                elem_to_shells.setdefault(int(sh.element), set()).add(sh.l)
            nelec = 0
            for z in numbers.tolist():
                val = _electron_configuration_valence_counts(int(z))
                present = elem_to_shells.get(int(z), set())
                nelec += int(round(sum(v for l, v in val.items() if l in present)))
            # Mode selection: EHT band (Γ-only or explicit k) vs k-SCF with atomic second-order
            if self._pbc_mode in ('eht-k', 'eht-gamma'):
                nocc = nelec // 2
                if nocc <= 0:
                    raise ValueError("Computed zero occupied bands; check basis/element configuration.")
                E_band = torch.tensor(0.0, dtype=positions.dtype, device=self.device)
                for Sk, Hk, wk in zip(Sks_raw, Hks, W):
                    evals_S, U = torch.linalg.eigh(Sk)
                    X = (U * evals_S.clamp_min(self._s_spd_floor).rsqrt()) @ U.conj().T
                    H_ortho = X.conj().T @ Hk @ X
                    evals, _ = torch.linalg.eigh(H_ortho)
                    E_band = E_band + wk.to(dtype=E_band.dtype) * evals.real[:nocc].sum()
            else:  # scf-k
                # Require Ewald parameters
                if self._ewald_eta is None or self._ewald_r_cut is None or self._ewald_g_cut is None:
                    raise ValueError("scf-k requires explicit ewald_eta, ewald_r_cut, ewald_g_cut (no defaults)")
                # Atomic-level γ^{(2)} under PBC via Ewald splitting
                from .pbc.second_order_pbc import compute_gamma2_atomic_pbc
                hub = map_hubbard_params(self._p_gxtb, self._schema)
                gamma2 = compute_gamma2_atomic_pbc(
                    numbers,
                    positions,
                    self._map_cn['r_cov'].to(device=self.device, dtype=positions.dtype),
                    hub['gamma'].to(device=self.device, dtype=positions.dtype),
                    cell,
                    ewald_eta=float(self._ewald_eta),
                    r_cut=float(self._ewald_r_cut),
                    g_cut=float(self._ewald_g_cut),
                )
                # Reference atomic charges: neutral valence counts per atom
                from .scf import _valence_electron_counts
                q_ref = _valence_electron_counts(numbers, basis)
                # AO->atom map consistent with S/H assembly above
                ao_atoms = blocks.get('ao_atoms', None)
                if ao_atoms is None:
                    ao_atoms_list: list[int] = []
                    for idx, sh in enumerate(basis.shells):
                        ao_atoms_list.extend([sh.atom_index] * basis.ao_counts[idx])
                    ao_atoms = torch.tensor(ao_atoms_list, dtype=torch.long, device=self.device)
                # Cache translations set for dynamic rebuilds
                translations0 = blocks.get('translations')
                if translations0 is not None:
                    logger.debug("Caching PBC translations: nR=%d", len(translations0))
                # Optional dynamic q-vSZP overlap under PBC SCF: rebuild S(k), H(k) each iteration
                k_builder = None
                if self._enable_dynamic_overlap:
                    # EEQBC charges required for q_eff (no hidden defaults)
                    info = getattr(atoms, "info", {}) or {}
                    q_info = info.get("q_eeqbc", None)
                    if q_info is None:
                        # Attempt to compute via internal EEQ if schema mapping is provided
                        eeq_map = getattr(self._schema, 'eeq', None)
                        if eeq_map is None:
                            raise ValueError("Dynamic q-vSZP under PBC requires atoms.info['q_eeqbc'] or [eeq] mapping in schema to compute EEQ charges.")
                        from .charges.eeq import compute_eeq_charges as _compute_eeq
                        mapping = {k: int(v) for k, v in eeq_map.items()}
                        q_eeqbc_t = _compute_eeq(numbers.to(device=self.device), positions.to(device=self.device), self._p_eeq, 0.0, mapping=mapping, device=self.device, dtype=positions.dtype)
                    else:
                        q_eeqbc_t = torch.tensor(q_info, dtype=positions.dtype, device=self.device)
                    # q-vSZP prefactors (k0..k3) from schema
                    qv = map_qvszp_prefactors(self._p_gxtb, self._schema)
                    k0 = qv['k0'].to(device=self.device, dtype=positions.dtype)
                    k1 = qv['k1'].to(device=self.device, dtype=positions.dtype)
                    k2 = qv['k2'].to(device=self.device, dtype=positions.dtype)
                    k3 = qv['k3'].to(device=self.device, dtype=positions.dtype)
                    r_cov = self._map_cn['r_cov'].to(device=self.device, dtype=positions.dtype)
                    k_cn = float(self._map_cn['k_cn'])
                    cn_cut = float(self._pbc_cn_cutoff)
                    # Capture constants for closure
                    # Shared tiny cache for coefficients and q_eff keyed by exact q bytes
                    coeff_cache: dict = {'key': None, 'coeffs': None, 'qeff': None}
                    coeff_cache_stats: dict = {'hits': 0, 'misses': 0}
                    def _coeffs_for_q(q_current: torch.Tensor):
                        qcur = q_current.to(dtype=positions.dtype, device=self.device).detach().cpu().numpy().tobytes()
                        if coeff_cache['key'] == qcur and coeff_cache['coeffs'] is not None:
                            coeff_cache_stats['hits'] += 1
                            logger.debug("qvSZP coeff cache HIT (iteration), key_bytes=%d", len(qcur))
                            return coeff_cache['coeffs'], coeff_cache['qeff']
                        q_eff = compute_effective_charge_pbc(numbers, positions, q_current.to(dtype=positions.dtype, device=self.device), q_eeqbc_t,
                                                             r_cov=r_cov, k_cn=k_cn, k0=k0, k1=k1, k2=k2, k3=k3,
                                                             cell=cell, cn_cutoff=cn_cut)
                        coeffs = build_dynamic_primitive_coeffs(numbers, basis, q_eff)
                        coeff_cache['key'] = qcur
                        coeff_cache['coeffs'] = coeffs
                        coeff_cache['qeff'] = q_eff
                        coeff_cache_stats['misses'] += 1
                        logger.debug("qvSZP coeff cache MISS → build coeffs (len=%d)", len(coeffs))
                        return coeffs, q_eff
                    def _k_builder(q_current: torch.Tensor):
                        # eq: doc/theory/7 Eq. (28) with PBC CN from doc/theory/25
                        coeffs, _ = _coeffs_for_q(q_current)
                        b = eht_lattice_blocks(numbers, positions, basis, self._p_gxtb, self._schema, cell, current_cut, float(self._pbc_cn_cutoff), coeff_override=coeffs, translations=translations0, ao_atoms_opt=ao_atoms)
                        m = assemble_k_matrices(b['translations'], b['S_blocks_raw'], b['H_blocks'], K)
                        return m['S_k'], m['H_k']
                    k_builder = _k_builder
                # Optional periodic AES builder (requires moment matrices)
                h_extra_builder = None
                if self._enable_aes:
                    if self._pbc_aes_cutoff is None or self._pbc_aes_cutoff <= 0:
                        raise ValueError("pbc_aes_cutoff must be provided (>0) to enable PBC AES.")
                    if self._pbc_cutoff is None:
                        raise ValueError("pbc_cutoff required for AES moments (S,D,Q) geometry consistency")
                    if self._pbc_cn_cutoff is None:
                        raise ValueError("pbc_cn_cutoff required for AES damping radii CN computation")
                    if not hasattr(self._schema, 'aes') or not hasattr(self._schema, 'aes_element'):
                        raise ValueError("AES requires schema [aes] and [aes.element] mappings")
                    from .hamiltonian.aes import AESParams as _AESParams
                    from .params.schema import map_aes_global, map_aes_element
                    aesg = map_aes_global(self._p_gxtb, self._schema)
                    aese = map_aes_element(self._p_gxtb, self._schema)
                    # Higher orders (n=7,9) not wired for PBC AES yet; ignore if present (no hidden defaults claimed)
                    aes_param_obj = _AESParams(
                        dmp3=float(aesg['dmp3']),
                        dmp5=float(aesg['dmp5']),
                        mprad=aese['mprad'].to(device=self.device, dtype=positions.dtype),
                        mpvcn=aese['mpvcn'].to(device=self.device, dtype=positions.dtype),
                        dmp7=float(aesg['dmp7']) if 'dmp7' in aesg else None,
                        dmp9=float(aesg['dmp9']) if 'dmp9' in aesg else None,
                    )
                    from .hamiltonian.moments_builder import build_moment_matrices
                    # Closure capturing periodic AES assembly from current Pk list and current atomic charges q
                    def _build_aes_from_Pk(Pk_list: list[torch.Tensor], q_current: torch.Tensor):
                        # Determine dynamic contraction coefficients if dynamic overlap is enabled
                        coeffs = None
                        if self._enable_dynamic_overlap:
                            # Reuse coefficients from the same q used for k-builder if available
                            coeffs, _ = _coeffs_for_q(q_current)
                        # Build AO moments with optional dynamic coefficients
                        S_mono, Dm, Qm = build_moment_matrices(numbers, positions, basis, coeff_override=coeffs)
                        # P_total = Σ_k w_k Pk
                        P_tot = torch.zeros_like(S_mono)
                        for Pk, wk in zip(Pk_list, W):
                            P_tot = P_tot + wk.to(dtype=P_tot.dtype) * Pk.to(dtype=P_tot.dtype)
                        from .pbc.aes_pbc import periodic_aes_potentials, assemble_aes_hamiltonian
                        v_mono, v_dip, v_quad, _E_pair = periodic_aes_potentials(
                            numbers, positions, basis, P_tot, S_mono, Dm, Qm, aes_param_obj,
                            r_cov=self._map_cn['r_cov'].to(device=self.device, dtype=positions.dtype),
                            k_cn=float(self._map_cn['k_cn']), cell=cell, cutoff=float(self._pbc_aes_cutoff),
                            ewald_eta=float(self._ewald_eta), ewald_r_cut=float(self._ewald_r_cut), ewald_g_cut=float(self._ewald_g_cut),
                            high_order_cutoff=(float(self._pbc_aes_high_order_cutoff) if self._pbc_aes_high_order_cutoff is not None else None),
                            si_rules=getattr(self._schema, 'aes_rules', None),
                        )
                        H_aes = assemble_aes_hamiltonian(S_mono, Dm, Qm, ao_atoms, v_mono, v_dip, v_quad)
                        # Double-counting consistent energy: E_dc = -1/2 Tr(H_AES P_tot)
                        E_dc = -0.5 * torch.einsum('ij,ji->', H_aes, P_tot)
                        # Coupling control: optionally decouple AES from SCF
                        if self._pbc_aes_couple:
                            H_list = [H_aes for _ in range(len(Sks_raw))]
                        else:
                            Z = torch.zeros_like(H_aes)
                            H_list = [Z for _ in range(len(Sks_raw))]
                        return H_list, E_dc
                    h_extra_builder = _build_aes_from_Pk
                from .pbc.scf_k import scf_k as _scf_k
                resk = _scf_k(numbers, basis, Sks_raw, Hks, ao_atoms, W.to(device=self.device, dtype=positions.dtype), nelec,
                              gamma2_atomic=gamma2, q_ref=q_ref.to(device=self.device, dtype=positions.dtype),
                              max_iter=self._scf_max_iter, tol=self._scf_tol, mix=self._scf_mix, h_extra_builder=h_extra_builder, k_builder=k_builder)
                # Report dynamic caches usage (debug)
                if self._enable_dynamic_overlap:
                    logger.debug("qvSZP coeff cache summary: hits=%d, misses=%d", coeff_cache_stats.get('hits', 0), coeff_cache_stats.get('misses', 0))
                E_band = resk.E_band
                if resk.E_extra is not None:
                    E_band = E_band + resk.E_extra
            # Optional D4 dispersion under PBC (two-body lattice sum; ATM not supported)
            if self._enable_dispersion:
                if self._pbc_disp_cutoff is None or self._pbc_disp_cutoff <= 0:
                    raise ValueError("pbc_disp_cutoff must be provided (>0) to enable PBC D4 dispersion.")
                from .classical.dispersion import load_d4_method
                from .classical.dispersion_pbc import d4_energy_pbc
                # Load method parameters
                method = load_d4_method(str(Path("parameters") / "dftd4parameters.toml"), variant=self._d4_variant, functional=self._d4_functional)
                # Build reference dataset or reuse cached, ensure it has periodic CN
                ref_static = self._d4_ref_static
                if ref_static is None:
                    ref_path = self._d4_reference_path or str(Path("parameters") / "d4_reference.toml")
                    from .params.loader import load_d4_reference_toml
                    ref_static = load_d4_reference_toml(ref_path, device=self.device, dtype=positions.dtype)
                    self._d4_ref_static = ref_static
                ref = dict(ref_static)
                ref['cn'] = cn  # periodic CN for current geometry
                E_disp = d4_energy_pbc(numbers, positions, charges, method, ref, cell, float(self._pbc_disp_cutoff))
                E_band = E_band + E_disp
            # Add classical zeroth-order terms under PBC as well: increment (elemental) and
            # semi-classical charge-dependent repulsion (doc/theory/10 and 11).
            # Use periodic CN computed above and EEQBC charges. If charges not provided, compute via internal EEQ
            # if schema defines mapping; otherwise raise to avoid hidden defaults.
            e_incr = energy_increment(numbers, self._deinc.to(device=self.device, dtype=positions.dtype))
            info = getattr(atoms, "info", {}) or {}
            q = info.get("q_eeqbc", None)
            if q is None:
                try:
                    from .charges.eeq import compute_eeq_charges as _compute_eeq
                    eeq_map = getattr(self._schema, 'eeq', None)
                    if eeq_map is None:
                        raise ValueError("Missing EEQ mapping in schema and atoms.info['q_eeqbc'] not provided.")
                    mapping = {k: int(v) for k, v in eeq_map.items()}
                    q_t = _compute_eeq(numbers.to(device=self.device), positions.to(device=self.device), self._p_eeq, 0.0, mapping=mapping, device=self.device, dtype=positions.dtype)
                    q = q_t.detach().cpu().numpy().tolist()
                except Exception as exc:
                    raise ValueError("PBC energy requires EEQBC charges for repulsion. Provide atoms.info['q_eeqbc'] or add [eeq] mapping to schema.") from exc
            charges = torch.tensor(q, dtype=positions.dtype, device=self.device)
            rep = RepulsionParams(
                z_eff0=self._map_rep["z_eff0"].to(device=self.device, dtype=positions.dtype),
                alpha0=self._map_rep["alpha0"].to(device=self.device, dtype=positions.dtype),
                kq=self._map_rep["kq"].to(device=self.device, dtype=positions.dtype),
                kq2=self._map_rep["kq2"].to(device=self.device, dtype=positions.dtype),
                kcn_elem=self._map_rep["kcn"].to(device=self.device, dtype=positions.dtype),
                r0=self._map_rep["r0"].to(device=self.device, dtype=positions.dtype),
                kpen1_hhe=float(self._map_rep["kpen1_hhe"]),
                kpen1_rest=float(self._map_rep["kpen1_rest"]),
                kpen2=float(self._map_rep["kpen2"]),
                kpen3=float(self._map_rep["kpen3"]),
                kpen4=float(self._map_rep["kpen4"]),
                kexp=float(self._map_rep["kexp"]),
                r_cov=self._map_cn["r_cov"].to(device=self.device, dtype=positions.dtype),
                k_cn=float(self._map_cn["k_cn"]),
            )
            e_rep = repulsion_energy(positions, numbers, rep, charges, cn=cn)
            E_total = E_band + e_incr + e_rep
            self.results["energy"] = float(E_total.item() * EH2EV)
            return

        # Increment energy (Eq. 50)
        e_incr = energy_increment(numbers, self._deinc.to(device=self.device, dtype=positions.dtype))

        # CN for repulsion exponent (Eq. 47/56)
        cn = coordination_number(
            positions,
            numbers,
            self._map_cn["r_cov"].to(device=self.device, dtype=positions.dtype),
            float(self._map_cn["k_cn"]),
        )

        # EEQBC charges: require explicit charges or compute via our EEQ model if schema mapping is provided
        info = getattr(atoms, "info", {}) or {}
        q = info.get("q_eeqbc", None)
        if q is None:
            # Compute charges using internal EEQ implementation if mapping is supplied via schema
            try:
                from .charges.eeq import compute_eeq_charges as _compute_eeq
                # Expect schema to provide mapping indices for eeq columns; if absent, raise
                eeq_map = getattr(self._schema, 'eeq', None)
                if eeq_map is None:
                    raise ValueError("Missing EEQ mapping in schema and atoms.info['q_eeqbc'] not provided.")
                # eeq_map should be dict-like with keys 'chi','eta','radius'
                mapping = {k: int(v) for k, v in eeq_map.items()}
                q_t = _compute_eeq(numbers.to(device=self.device), positions.to(device=self.device), self._p_eeq, 0.0, mapping=mapping, device=self.device, dtype=positions.dtype)
                q = q_t.detach().cpu().numpy().tolist()
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    "EEQBC charges missing. Provide atoms.info['q_eeqbc'] or add [eeq] mapping (chi, eta, radius) to schema."
                ) from exc
        charges = torch.tensor(q, dtype=positions.dtype, device=self.device)

        rep = RepulsionParams(
            z_eff0=self._map_rep["z_eff0"].to(device=self.device, dtype=positions.dtype),
            alpha0=self._map_rep["alpha0"].to(device=self.device, dtype=positions.dtype),
            kq=self._map_rep["kq"].to(device=self.device, dtype=positions.dtype),
            kq2=self._map_rep["kq2"].to(device=self.device, dtype=positions.dtype),
            kcn_elem=self._map_rep["kcn"].to(device=self.device, dtype=positions.dtype),
            r0=self._map_rep["r0"].to(device=self.device, dtype=positions.dtype),
            kpen1_hhe=float(self._map_rep["kpen1_hhe"]),
            kpen1_rest=float(self._map_rep["kpen1_rest"]),
            kpen2=float(self._map_rep["kpen2"]),
            kpen3=float(self._map_rep["kpen3"]),
            kpen4=float(self._map_rep["kpen4"]),
            kexp=float(self._map_rep["kexp"]),
            r_cov=self._map_cn["r_cov"].to(device=self.device, dtype=positions.dtype),
            k_cn=float(self._map_cn["k_cn"]),
        )

        e_rep = repulsion_energy(positions, numbers, rep, charges, cn=cn)

        # Electronic energy via SCF (EHT core); closed-shell by default
        basis = build_atom_basis(numbers, self._p_basis)
        core = build_eht_core(numbers, positions, basis, self._p_gxtb, self._schema)
        builder = make_core_builder(basis, self._p_gxtb, self._schema)
        # Hubbard params
        hub = map_hubbard_params(self._p_gxtb, self._schema)
        # AO -> atom
        ao_atoms = core['ao_atoms']
        # Number of valence electrons from electron configuration heuristic (Eq. 99 context)
        # Determine electrons consistent with basis shells present for each element
        elem_to_shells: Dict[int, set[str]] = {}
        for sh in basis.shells:
            elem_to_shells.setdefault(int(sh.element), set()).add(sh.l)
        nelec = 0
        for z in numbers.tolist():
            val = _electron_configuration_valence_counts(int(z))
            present = elem_to_shells.get(int(z), set())
            nelec += int(round(sum(v for l, v in val.items() if l in present)))
        # Enable shell-resolved second-order TB (doc/theory/15, Eqs. 98–106) and prepare shell params
        # Build ShellSecondOrderParams from element Hubbard gamma as U^{(2),0} baseline and provide CN.
        so_params = None
        shell_params_for_orders = None
        try:
            if self._enable_second_order or self._enable_third_order:
                from .hamiltonian.second_order_tb import build_shell_second_order_params as _build_sp
                # Guard: require strictly positive Hubbard gamma for all present elements (no hidden defaults)
                gvec = hub['gamma']
                g_present = gvec[numbers.long()]
                if torch.any(g_present <= 0):
                    bad = sorted(set(int(z) for z, gv in zip(numbers.tolist(), g_present.tolist()) if gv <= 0))
                    raise ValueError(
                        f"Schema [hubbard].gamma is non-positive for elements Z={bad}; cannot enable shell second order (doc/theory/15). "
                        "Fix gxtb.schema.toml mapping for gamma or disable second_order."
                    )
                shell_params_for_orders = _build_sp(int(numbers.max().item()), gvec)
            if self._enable_second_order and shell_params_for_orders is not None:
                # Reuse CN computed above for current geometry
                so_params = {  # type: ignore[assignment]
                    'shell_params': shell_params_for_orders,
                    'cn': cn,
                }
        except Exception as exc:
            so_params = None
            shell_params_for_orders = None
        # Run SCF with second-order shell corrections enabled (doc/theory/15)
        # q‑vSZP dynamic overlap parameters (doc/theory/7, Eq. 28) from schema
        if self._enable_dynamic_overlap:
            qv_pref = map_qvszp_prefactors(self._p_gxtb, self._schema)
            qv_pack = {
                'k0': qv_pref['k0'].to(device=self.device, dtype=positions.dtype),
                'k1': qv_pref['k1'].to(device=self.device, dtype=positions.dtype),
                'k2': qv_pref['k2'].to(device=self.device, dtype=positions.dtype),
                'k3': qv_pref['k3'].to(device=self.device, dtype=positions.dtype),
                'r_cov': self._map_cn['r_cov'].to(device=self.device, dtype=positions.dtype),
                'k_cn': float(self._map_cn['k_cn']),
            }
        else:
            qv_pack = None
        # Optional AES parameters
        aes_params_dict = None
        if self._enable_aes:
            try:
                from .hamiltonian.aes import AESParams as _AESParams
                from .params.schema import map_aes_global, map_aes_element
                aesg = map_aes_global(self._p_gxtb, self._schema)
                aese = map_aes_element(self._p_gxtb, self._schema)
                aes_param_obj = _AESParams(
                    dmp3=float(aesg['dmp3']),
                    dmp5=float(aesg['dmp5']),
                    mprad=aese['mprad'].to(device=self.device, dtype=positions.dtype),
                    mpvcn=aese['mpvcn'].to(device=self.device, dtype=positions.dtype),
                    dmp7=float(aesg['dmp7']) if 'dmp7' in aesg else None,
                    dmp9=float(aesg['dmp9']) if 'dmp9' in aesg else None,
                )
                aes_params_dict = {
                    'params': aes_param_obj,
                    'r_cov': self._map_cn['r_cov'].to(device=self.device, dtype=positions.dtype),
                    'k_cn': float(self._map_cn['k_cn']),
                    'si_rules': getattr(self._schema, 'aes_rules', None),
                }
            except Exception:
                aes_params_dict = None
        # Optional third/fourth order parameters
        third_shell_params = None
        third_params = None
        if self._enable_third_order:
            if shell_params_for_orders is None:
                # build if not built above
                from .hamiltonian.second_order_tb import build_shell_second_order_params as _build_sp
                shell_params_for_orders = _build_sp(int(numbers.max().item()), hub['gamma'])
            if shell_params_for_orders is not None:
                third_shell_params = {  # type: ignore[assignment]
                    'shell_params': shell_params_for_orders,
                    'cn': cn,
                }
                from .params.schema import map_third_order_params
                tp = map_third_order_params(self._p_gxtb, self._schema)
                third_params = {
                    'gamma3_elem': tp['gamma3_elem'].to(device=self.device, dtype=positions.dtype),
                    'kGamma': tp['kGamma'].to(device=self.device, dtype=positions.dtype),
                    'k3': float(tp['k3']),
                    'k3x': float(tp['k3x']),
                }
        fourth_params = None
        if self._enable_fourth_order:
            from .params.schema import map_fourth_order_params
            fourth_params = {'gamma4': float(map_fourth_order_params(self._p_gxtb, self._schema))}

        # Warm-start initial charges for SCF: prefer last SCF q if available and numbers unchanged
        init_q = charges
        if getattr(self, '_last_scf_q', None) is not None and getattr(self, '_last_scf_numbers', None) is not None:
            try:
                if torch.equal(self._last_scf_numbers.to(device=self.device), numbers.to(device=self.device)) and self._last_scf_q.shape == charges.shape:
                    init_q = self._last_scf_q.to(device=self.device, dtype=positions.dtype)
            except Exception:
                init_q = charges

        try:
            # Increment SCF cycle counter for logging
            self._scf_call_counter += 1
            res = scf(
                numbers,
                positions,
                basis,
                builder,
                core['S'],
                hubbard=hub,
                ao_atoms=ao_atoms,
                nelec=nelec,
                max_iter=self._scf_max_iter,
                tol=self._scf_tol,
                mix=self._scf_mix,
                mixing={
                    'scheme': self._scf_mixing,
                    'beta': self._scf_mix,
                    'history': self._scf_mixing_history,
                    'beta_init': self._scf_mix_init,
                    'anderson_soft_start': self._scf_anderson_soft_start,
                    'anderson_diag_offset': self._scf_anderson_diag_offset,
                    'beta_min': self._scf_beta_min,
                    'beta_max': self._scf_beta_max,
                    'beta_decay': self._scf_beta_decay,
                    'restart_on_nan': self._scf_restart_on_nan,
                },
                second_order=self._enable_second_order,
                so_params=so_params if so_params is not None else {},
                third_order=self._enable_third_order,
                third_shell_params=third_shell_params,
                third_params=third_params,
                fourth_order=self._enable_fourth_order,
                fourth_params=fourth_params,
                eeq_charges=init_q,
                dynamic_overlap=bool(qv_pack is not None),
                qvszp_params=qv_pack if qv_pack is not None else None,
                aes=self._enable_aes and (aes_params_dict is not None),
                aes_params=aes_params_dict,
                uhf=self._uhf,
                nelec_alpha=self._nelec_alpha,
                nelec_beta=self._nelec_beta,
                scp_mode=self._scf_scp_mode,
                log_cycle=self._scf_call_counter,
            )
        except Exception as exc:
            # Shell-resolved second-order may fail for elements whose valence includes shells
            # absent in the basis (doc/theory/15, Eq. 99 reference populations). In that case,
            # fall back to atomic second-order path (Eqs. 100b–106) with η and r_cov, which
            # does not require shell reference populations.
            cn_map = self._map_cn
            so_atomic = {  # type: ignore[assignment]
                'eta': hub['gamma'],
                'r_cov': cn_map['r_cov'],
            }
            self._scf_call_counter += 1
            res = scf(
                numbers,
                positions,
                basis,
                builder,
                core['S'],
                hubbard=hub,
                ao_atoms=ao_atoms,
                nelec=nelec,
                max_iter=self._scf_max_iter,
                tol=self._scf_tol,
                 mix=self._scf_mix,
                 mixing={
                    'scheme': self._scf_mixing,
                    'beta': self._scf_mix,
                    'history': self._scf_mixing_history,
                    'beta_init': self._scf_mix_init,
                    'anderson_soft_start': self._scf_anderson_soft_start,
                    'anderson_diag_offset': self._scf_anderson_diag_offset,
                    'beta_min': self._scf_beta_min,
                    'beta_max': self._scf_beta_max,
                    'beta_decay': self._scf_beta_decay,
                    'restart_on_nan': self._scf_restart_on_nan,
                 },
                second_order=self._enable_second_order,
                so_params=so_atomic,
                third_order=self._enable_third_order,
                third_shell_params=third_shell_params,
                third_params=third_params,
                fourth_order=self._enable_fourth_order,
                fourth_params=fourth_params,
                eeq_charges=init_q,
                dynamic_overlap=bool(qv_pack is not None),
                qvszp_params=qv_pack if qv_pack is not None else None,
                aes=self._enable_aes and (aes_params_dict is not None),
                aes_params=aes_params_dict,
                uhf=self._uhf,
                nelec_alpha=self._nelec_alpha,
                nelec_beta=self._nelec_beta,
                scp_mode=self._scf_scp_mode,
                log_cycle=self._scf_call_counter,
            )
        e_el = res.E_elec if res.E_elec is not None else torch.einsum('ij,ji->', res.P, core['H0'])
        e_total = (e_incr + e_rep + e_el)

        # Optional D4 dispersion energy addition (no SCF coupling)
        if self._enable_dispersion:
            from .classical.dispersion import load_d4_method, d4_energy
            # Load method parameters from TOML using our loader semantics
            method = load_d4_method(str(Path("parameters") / "dftd4parameters.toml"), variant=self._d4_variant, functional=self._d4_functional)
            # Build reference dataset according to explicit source controls
            ref_static = self._d4_ref_static
            if ref_static is None:
                # Load from TOML reference
                ref_path = self._d4_reference_path or str(Path("parameters") / "d4_reference.toml")
                from .params.loader import load_d4_reference_toml
                ref_static = load_d4_reference_toml(ref_path, device=self.device, dtype=positions.dtype)
                self._d4_ref_static = ref_static
            # Compose per-geometry ref by attaching CN (consistent with our CN model)
            ref = dict(ref_static)
            # Sanity: ensure all elements supported by the D4 reference dataset
            zs = set(numbers.long().tolist())
            z_sup = set([int(z) for z in ref_static.get('z_supported', torch.tensor([], device=self.device)).tolist()])
            if not zs.issubset(z_sup):
                missing = sorted(list(zs.difference(z_sup)))
                raise ValueError(
                    f"D4 reference dataset missing elements Z={missing}. Extend parameters/d4_reference.toml or pass d4_ref_dict."
                )
            ref['cn'] = cn  # reuse CN computed above for repulsion
            # Compute dispersion energy (Hartree), add to total
            E_disp = d4_energy(numbers, positions, charges, method, ref)
            e_total = e_total + E_disp
        # Store energy and a per-term breakdown (eV) for diagnostics
        self.results["energy"] = float(e_total.item() * EH2EV)
        self.results["E_increment_eV"] = float(e_incr.item() * EH2EV)
        self.results["E_repulsion_eV"] = float(e_rep.item() * EH2EV)
        self.results["E_elec_eV"] = float(e_el.item() * EH2EV)
        if res.E2 is not None:
            try:
                self.results["E2_eV"] = float(res.E2.item() * EH2EV)
            except Exception:
                self.results["E2_eV"] = None
        else:
            self.results["E2_eV"] = None
        try:
            self.results["q"] = res.q.detach().cpu().tolist()
        except Exception:
            self.results["q"] = None
        # Update warm-start cache for forces
        try:
            self._last_scf_q = res.q.detach().clone()
            self._last_scf_numbers = numbers.detach().clone().to(dtype=torch.long, device=self.device)
        except Exception:
            self._last_scf_q = None
            self._last_scf_numbers = None
        # Optional Molden output (non-PBC only)
        if self._molden_path:
            # Build shells with dynamic contractions matching SCF settings
            if qv_pack is not None:
                # q_eff per doc/theory/7 Eq. (28)
                q_eff = compute_effective_charge(
                    numbers,
                    positions,
                    res.q.to(device=self.device, dtype=positions.dtype),
                    charges.to(device=self.device, dtype=positions.dtype),
                    r_cov=self._map_cn['r_cov'],
                    k_cn=float(self._map_cn['k_cn']),
                    k0=qv_pack['k0'], k1=qv_pack['k1'], k2=qv_pack['k2'], k3=qv_pack['k3'],
                ).cpu().tolist()
            else:
                # Static basis contractions (c = c0)
                q_eff = [0.0] * len(numbers)
            shells = shells_from_qvszp(numbers.cpu().tolist(), self._p_basis, q_eff)
            # AO row scaling per option
            ao_row_scale = None
            if self._molden_norm in ('sqrt_sii', 'inv_sqrt_sii'):
                Sii = torch.diag(res.S if res.S is not None else core['S']).detach().cpu()
                if self._molden_norm == 'sqrt_sii':
                    ao_row_scale = [float(x.sqrt().item()) for x in Sii]
                else:
                    ao_row_scale = [float((x.clamp_min(1e-30).rsqrt()).item()) for x in Sii]
            # Build MO sets
            if self._uhf and (res.C_alpha is not None and res.C_beta is not None and res.eps_alpha is not None and res.eps_beta is not None):
                # Determine spin occupations
                if self._nelec_alpha is None or self._nelec_beta is None:
                    val_e = _scf_mod._valence_electron_counts(numbers, build_atom_basis(numbers, self._p_basis)).to(device=self.device)
                    nelec = int(round(float(val_e.sum().item())))
                    na = (nelec + 1) // 2
                    nb = nelec // 2
                else:
                    na, nb = int(self._nelec_alpha), int(self._nelec_beta)
                occ_a = [1.0 if i < na else 0.0 for i in range(int(res.C_alpha.shape[1]))]
                occ_b = [1.0 if i < nb else 0.0 for i in range(int(res.C_beta.shape[1]))]
                wf = MOWavefunction(
                    alpha=MOSet(coeff=res.C_alpha.detach().cpu().tolist(), energy=[float(x) for x in res.eps_alpha.detach().cpu().tolist()], occ=occ_a),
                    beta=MOSet(coeff=res.C_beta.detach().cpu().tolist(), energy=[float(x) for x in res.eps_beta.detach().cpu().tolist()], occ=occ_b),
                )
            else:
                # RHF occupations
                val_e = _scf_mod._valence_electron_counts(numbers, build_atom_basis(numbers, self._p_basis)).to(device=self.device)
                nelec = int(round(float(val_e.sum().item())))
                nocc = nelec // 2
                nmo = int(res.C.shape[1])
                occ = [2.0 if i < nocc else 0.0 for i in range(nmo)]
                wf = MOWavefunction(
                    alpha=MOSet(coeff=res.C.detach().cpu().tolist(), energy=[float(x) for x in res.eps.cpu().tolist()], occ=occ)
                )
            symbols = getattr(atoms, 'get_chemical_symbols')() if hasattr(atoms, 'get_chemical_symbols') else [str(int(z)) for z in numbers.tolist()]
            coords = atoms.get_positions().tolist()
            write_molden(
                self._molden_path,
                numbers=numbers.cpu().tolist(),
                symbols=symbols,
                coords=coords,
                unit="Angs",
                shells=shells,
                wf=wf,
                spherical=self._molden_spherical,
                program="gxtb",
                version=None,
                ao_row_scale=ao_row_scale,
            )
        
    def get_forces(self, atoms):  # type: ignore
        """Return forces (eV/Å): analytic (non-PBC) or numeric via finite differences.

        - Analytic mode aggregates per-term gradients implemented in gxtb.grad.nuclear (doc/theory/6).
          Only supported for non-PBC geometries. Raises if required parameters are missing.
        - Numeric mode supports central (default) and forward differences.
        """
        # Determine step and scheme
        try:
            pbc_flags = atoms.get_pbc()
        except Exception:
            pbc_flags = (False, False, False)
        # Analytic forces path (non-PBC only)
        if not any(bool(x) for x in tuple(pbc_flags)) and getattr(self, "_force_mode", "numeric") == "analytic":
            # Ensure latest SCF state matches current geometry
            _ = self.get_potential_energy(atoms)
            numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int64, device=self.device)
            positions = torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype(), device=self.device)
            basis = build_atom_basis(numbers, self._p_basis)
            # Build repulsion params (for gradient) and EEQ charges + derivative
            from .params.schema import map_repulsion_params, map_cn_params
            from .classical.repulsion import RepulsionParams as _RParams
            rp = map_repulsion_params(self._p_gxtb, self._schema)
            cnm = map_cn_params(self._p_gxtb, self._schema)
            rep_params = _RParams(
                z_eff0=rp['z_eff0'].to(device=self.device, dtype=positions.dtype),
                alpha0=rp['alpha0'].to(device=self.device, dtype=positions.dtype),
                kq=rp['kq'].to(device=self.device, dtype=positions.dtype),
                kq2=rp['kq2'].to(device=self.device, dtype=positions.dtype),
                kcn_elem=rp['kcn'].to(device=self.device, dtype=positions.dtype),
                r0=rp['r0'].to(device=self.device, dtype=positions.dtype),
                kpen1_hhe=float(rp['kpen1_hhe']),
                kpen1_rest=float(rp['kpen1_rest']),
                kpen2=float(rp['kpen2']),
                kpen3=float(rp['kpen3']),
                kpen4=float(rp['kpen4']),
                kexp=float(rp['kexp']),
                r_cov=cnm['r_cov'].to(device=self.device, dtype=positions.dtype),
                k_cn=float(cnm['k_cn']),
            )
            # EEQ charges and derivative for repulsion Eq. 58
            from .charges.eeq import compute_eeq_charges, compute_eeq_charge_derivative
            eeq_map = getattr(self._schema, 'eeq', None)
            if eeq_map is None:
                raise ValueError("Analytic forces require [eeq] schema mapping for charges/derivatives")
            mapping = {k: int(v) for k, v in eeq_map.items()}
            q_eeq = compute_eeq_charges(numbers, positions, self._p_eeq, 0.0, mapping=mapping, device=self.device, dtype=positions.dtype)
            dq_dpos = compute_eeq_charge_derivative(numbers, positions, self._p_eeq, 0.0, mapping=mapping, device=self.device, dtype=positions.dtype)
            # SCF to get density and charges (reuse energy path configuration)
            core = build_eht_core(numbers, positions, basis, self._p_gxtb, self._schema)
            builder = make_core_builder(basis, self._p_gxtb, self._schema)
            # Hubbard params & second/third/fourth settings as in energy path
            hub = map_hubbard_params(self._p_gxtb, self._schema)
            # Electron count
            elem_to_shells: Dict[int, set[str]] = {}
            for sh in basis.shells:
                elem_to_shells.setdefault(int(sh.element), set()).add(sh.l)
            nelec = 0
            for z in numbers.tolist():
                val = _electron_configuration_valence_counts(int(z))
                present = elem_to_shells.get(int(z), set())
                nelec += int(round(sum(v for l, v in val.items() if l in present)))
            # Second/third/fourth and AES params mirroring calculate()
            so_params = None
            shell_params_for_orders = None
            try:
                if self._enable_second_order or self._enable_third_order:
                    from .hamiltonian.second_order_tb import build_shell_second_order_params as _build_sp
                    gvec = hub['gamma']
                    shell_params_for_orders = _build_sp(int(numbers.max().item()), gvec)
                if self._enable_second_order and shell_params_for_orders is not None:
                    from .cn import coordination_number
                    cn = coordination_number(positions, numbers, cnm['r_cov'].to(device=self.device, dtype=positions.dtype), float(cnm['k_cn']))
                    so_params = {'shell_params': shell_params_for_orders, 'cn': cn}
            except Exception:
                so_params = None
            aes_params_dict = None
            if self._enable_aes:
                try:
                    from .hamiltonian.aes import AESParams as _AESParams
                    from .params.schema import map_aes_global, map_aes_element
                    aesg = map_aes_global(self._p_gxtb, self._schema)
                    aese = map_aes_element(self._p_gxtb, self._schema)
                    aes_params_dict = {
                        'params': _AESParams(
                            dmp3=float(aesg['dmp3']),
                            dmp5=float(aesg['dmp5']),
                            mprad=aese['mprad'].to(device=self.device, dtype=positions.dtype),
                            mpvcn=aese['mpvcn'].to(device=self.device, dtype=positions.dtype),
                            dmp7=float(aesg['dmp7']) if 'dmp7' in aesg else None,
                            dmp9=float(aesg['dmp9']) if 'dmp9' in aesg else None,
                        ),
                        'r_cov': cnm['r_cov'].to(device=self.device, dtype=positions.dtype),
                        'k_cn': float(cnm['k_cn']),
                        'si_rules': getattr(self._schema, 'aes_rules', None),
                    }
                except Exception:
                    aes_params_dict = None
            # Third/fourth order params
            third_shell_params = None
            third_params = None
            if self._enable_third_order:
                if shell_params_for_orders is None:
                    from .hamiltonian.second_order_tb import build_shell_second_order_params as _build_sp
                    shell_params_for_orders = _build_sp(int(numbers.max().item()), hub['gamma'])
                if shell_params_for_orders is not None:
                    third_shell_params = {'shell_params': shell_params_for_orders, 'cn': cn}
                    from .params.schema import map_third_order_params
                    tp = map_third_order_params(self._p_gxtb, self._schema)
                    third_params = {
                        'gamma3_elem': tp['gamma3_elem'].to(device=self.device, dtype=positions.dtype),
                        'kGamma': tp['kGamma'].to(device=self.device, dtype=positions.dtype),
                        'k3': float(tp['k3']),
                        'k3x': float(tp['k3x']),
                    }
            fourth_params = None
            if self._enable_fourth_order:
                from .params.schema import map_fourth_order_params
                fourth_params = {'gamma4': float(map_fourth_order_params(self._p_gxtb, self._schema))}

            # Run SCF to get P and q (with fallback to atomic second-order like calculate())
            try:
                self._scf_call_counter += 1
                res = scf(
                    numbers, positions, basis, builder, core['S'],
                    hubbard=hub, ao_atoms=core['ao_atoms'], nelec=nelec,
                    max_iter=self._scf_max_iter, tol=self._scf_tol, mix=self._scf_mix,
                    mixing={
                        'scheme': self._scf_mixing,
                        'beta': self._scf_mix,
                        'history': self._scf_mixing_history,
                        'beta_init': self._scf_mix_init,
                        'anderson_soft_start': self._scf_anderson_soft_start,
                        'anderson_diag_offset': self._scf_anderson_diag_offset,
                        'beta_min': self._scf_beta_min,
                        'beta_max': self._scf_beta_max,
                        'beta_decay': self._scf_beta_decay,
                        'restart_on_nan': self._scf_restart_on_nan,
                    },
                    second_order=self._enable_second_order, so_params=(so_params if so_params is not None else {}),
                    third_order=self._enable_third_order, third_shell_params=third_shell_params, third_params=third_params,
                    fourth_order=self._enable_fourth_order, fourth_params=fourth_params,
                    eeq_charges=q_eeq,
                    scp_mode=self._scf_scp_mode,
                    log_cycle=self._scf_call_counter,
                )
            except Exception:
                # Fallback: atomic second-order only
                so_atomic = {'eta': hub['gamma'], 'r_cov': cnm['r_cov']}
                self._scf_call_counter += 1
                res = scf(
                    numbers, positions, basis, builder, core['S'],
                    hubbard=hub, ao_atoms=core['ao_atoms'], nelec=nelec,
                    max_iter=self._scf_max_iter, tol=self._scf_tol, mix=self._scf_mix,
                    mixing={
                        'scheme': self._scf_mixing,
                        'beta': self._scf_mix,
                        'history': self._scf_mixing_history,
                        'beta_init': self._scf_mix_init,
                        'anderson_soft_start': self._scf_anderson_soft_start,
                        'anderson_diag_offset': self._scf_anderson_diag_offset,
                        'beta_min': self._scf_beta_min,
                        'beta_max': self._scf_beta_max,
                        'beta_decay': self._scf_beta_decay,
                        'restart_on_nan': self._scf_restart_on_nan,
                    },
                    second_order=self._enable_second_order, so_params=so_atomic,
                    third_order=False, fourth_order=self._enable_fourth_order, fourth_params=fourth_params,
                    eeq_charges=q_eeq,
                    scp_mode=self._scf_scp_mode,
                    log_cycle=self._scf_call_counter,
                )
            P = res.P
            # Dispersion parameters (optional)
            disp_params = None
            if self._enable_dispersion:
                from .classical.dispersion import load_d4_method
                method = load_d4_method(str(Path("parameters") / "dftd4parameters.toml"), variant=self._d4_variant, functional=self._d4_functional)
                ref_static = self._d4_ref_static
                if ref_static is None:
                    ref_path = self._d4_reference_path or str(Path("parameters") / "d4_reference.toml")
                    from .params.loader import load_d4_reference_toml
                    ref_static = load_d4_reference_toml(ref_path, device=self.device, dtype=positions.dtype)
                    self._d4_ref_static = ref_static
                ref = dict(ref_static)
                # Attach CN (molecular)
                from .cn import coordination_number
                ref['cn'] = coordination_number(positions, numbers, cnm['r_cov'].to(device=self.device, dtype=positions.dtype), float(cnm['k_cn']))
                disp_params = {'method': method, 'ref': ref, 'q': q_eeq}
            # Total gradient aggregation
            from .grad.nuclear import total_gradient as _totgrad
            g = _totgrad(
                numbers, positions, basis, self._p_gxtb, self._schema,
                P=P,
                include_eht_stepA=True,
                include_dynamic_overlap_cn=bool(self._enable_dynamic_overlap),
                q_scf=(res.q if hasattr(res, 'q') and res.q is not None else None),
                q_eeqbc=q_eeq,
                include_second_order=bool(self._enable_second_order),
                # atomic isotropic second-order for gradients (shell-resolved gradient not yet wired)
                so_params={'eta': hub['gamma'], 'r_cov': cnm['r_cov']},
                q=(res.q if hasattr(res, 'q') else None),
                q_ref=(res.q_ref if hasattr(res, 'q_ref') else None),
                include_aes=bool(self._enable_aes and (aes_params_dict is not None)),
                aes_params=(aes_params_dict['params'] if aes_params_dict is not None else None),
                aes_r_cov=(aes_params_dict['r_cov'] if aes_params_dict is not None else None),
                aes_k_cn=(aes_params_dict['k_cn'] if aes_params_dict is not None else None),
                include_repulsion=True,
                repulsion_params=rep_params,
                eeq=self._p_eeq,
                repulsion_dq_dpos=dq_dpos,
                include_dispersion=bool(self._enable_dispersion),
                dispersion_params=disp_params,
            )
            # Convert to forces (eV/Å)
            f = (-g * EH2EV).detach().cpu().numpy()
            return f
        # Base epsilon on PBC smoothness rule; allow override via self._force_eps
        base_eps = 3.0e-4 if any(bool(x) for x in tuple(pbc_flags)) else 1.0e-3
        eps = float(getattr(self, "_force_eps", base_eps))
        scheme = getattr(self, "_force_diff", "central")
        # Freeze SPD adaptation under PBC to keep lattice block set identical for ±eps displacements
        orig_adapt = getattr(self, "_s_psd_adapt", False)
        if any(bool(x) for x in tuple(pbc_flags)):
            self._s_psd_adapt = False
        # Enable warm-start during force evaluation
        prev_force_flag = getattr(self, '_force_eval_mode', False)
        self._force_eval_mode = True
        # Optionally suppress SCF INFO logs during finite differences
        scf_logger = logging.getLogger('gxtb.scf')
        prev_level = scf_logger.level
        if not getattr(self, '_force_log', False):
            scf_logger.setLevel(logging.WARNING)
        # Force accumulation
        f = []
        pos = atoms.get_positions()
        # For forward differences, get baseline energy once
        if scheme == 'forward':
            atoms.set_positions(pos)
            e0 = self.get_potential_energy(atoms)
        for a in range(len(atoms)):
            f_comp = [0.0, 0.0, 0.0]
            for k in range(3):
                if scheme == 'central':
                    dp = pos.copy(); dp[a, k] += eps
                    atoms.set_positions(dp)
                    ep = self.get_potential_energy(atoms)
                    dm = pos.copy(); dm[a, k] -= eps
                    atoms.set_positions(dm)
                    em = self.get_potential_energy(atoms)
                    f_comp[k] = - (ep - em) / (2 * eps)
                else:  # forward
                    dp = pos.copy(); dp[a, k] += eps
                    atoms.set_positions(dp)
                    ep = self.get_potential_energy(atoms)
                    f_comp[k] = - (ep - e0) / eps
            f.append(f_comp)
        atoms.set_positions(pos)
        # Restore warm-start and SPD adaptation flags and logger level
        self._force_eval_mode = prev_force_flag
        self._s_psd_adapt = orig_adapt
        try:
            scf_logger.setLevel(prev_level)
        except Exception:
            pass
        return torch.tensor(f, dtype=torch.get_default_dtype()).cpu().numpy()
