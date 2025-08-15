from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

import torch

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
from .basis.qvszp import build_atom_basis
from .hamiltonian.scf_adapter import build_eht_core, make_core_builder
from .scf import scf
from .hamiltonian.second_order_tb import _electron_configuration_valence_counts

EH2EV = 27.211386245988


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
        enable_dynamic_overlap: bool = True,
        enable_aes: bool = False,
        # Dispersion (DFT-D4) controls
        enable_dispersion: bool = False,
        d4_variant: str = "bj-eeq-atm",
        d4_functional: Optional[str] = None,
        d4_reference_path: Optional[str] = None,
        d4_ref_dict: Optional[dict] = None,
        # PBC controls (see doc/theory/25_periodic_boundary_conditions.md)
        pbc_mode: Optional[str] = None,  # None, 'eht-gamma', 'eht-k', 'scf-k' (future)
        kpoints: Optional[list] = None,  # list of fractional k-points [[kx,ky,kz], ...]
        kpoint_weights: Optional[list] = None,  # same length as kpoints; sum to 1.0
        pbc_cutoff: Optional[float] = None,  # real-space cutoff (Angstrom) for S/H blocks
        pbc_cn_cutoff: Optional[float] = None,  # cutoff for CN under PBC
        pbc_disp_cutoff: Optional[float] = None,  # cutoff for dispersion under PBC
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
        self._enable_dynamic_overlap = bool(enable_dynamic_overlap)
        self._enable_aes = bool(enable_aes)
        # Dispersion switches (no hidden defaults)
        self._enable_dispersion = bool(enable_dispersion)
        self._d4_variant = str(d4_variant)
        self._d4_functional = d4_functional
        self._d4_reference_path = d4_reference_path
        self._d4_ref_static = d4_ref_dict  # geometry-independent tensors; we'll attach 'cn' per geometry
        # PBC configuration
        self._pbc_mode = pbc_mode
        self._kpoints = kpoints
        self._kpoint_weights = kpoint_weights
        self._pbc_cutoff = pbc_cutoff
        self._pbc_cn_cutoff = pbc_cn_cutoff
        self._pbc_disp_cutoff = pbc_disp_cutoff

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
            # Ensure explicit configuration was provided; no hidden defaults allowed.
            if self._pbc_mode is None:
                raise NotImplementedError(
                    "PBC requested by ASE Atoms.pbc, but no PBC mode configured. "
                    "Pass pbc_mode='eht-k' with explicit kpoints/kpoint_weights and pbc_cutoff as per doc/theory/25_periodic_boundary_conditions.md."
                )
            # For now, only EHT Bloch sum path is allowed; SCF across k requires Ewald for second-order terms.
            if self._pbc_mode not in ("eht-gamma", "eht-k"):
                raise NotImplementedError(
                    f"Unsupported pbc_mode={self._pbc_mode!r}. Supported: 'eht-gamma' (Γ-only band energy) or 'eht-k' (general k)."
                )
            # Validate required inputs are present and explicit
            if self._pbc_mode == "eht-k":
                if not isinstance(self._kpoints, list) or not isinstance(self._kpoint_weights, list):
                    raise ValueError(
                        "kpoints and kpoint_weights must be provided as lists for PBC runs (no implicit grids)."
                    )
                if abs(sum(float(w) for w in self._kpoint_weights) - 1.0) > 1e-12:
                    raise ValueError("Sum of kpoint_weights must equal 1.0 exactly (within numerical tolerance).")
            # Γ-only mode does not require kpoint lists; it implies k=[0,0,0], w=1 explicitly
            if self._pbc_cutoff is None or self._pbc_cutoff <= 0:
                raise ValueError("pbc_cutoff (real-space cutoff in Angstrom for S/H blocks) must be > 0.")
            if self._pbc_cn_cutoff is None or self._pbc_cn_cutoff <= 0:
                raise ValueError("pbc_cn_cutoff (cutoff for CN under PBC) must be > 0.")
            # Disallow unsupported terms explicitly (no hidden defaults)
            if self._enable_second_order:
                raise NotImplementedError("second_order=True not supported under PBC; isotropic electrostatics requires Ewald (see doc/theory/25)")
            if self._enable_aes:
                raise NotImplementedError("AES under PBC requires explicit real-space cutoff and tail control; disable enable_aes or provide a dedicated PBC path.")
            if self._enable_dispersion:
                raise NotImplementedError("DFT-D4 dispersion under PBC not wired yet; disable enable_dispersion or provide R_disp and periodic machinery.")

            # Assemble Bloch-sum EHT band energy (no SCF; doc/theory/25)
            from .pbc.cell import validate_cell
            from .pbc.kpoints import validate_kpoints
            from .pbc.bloch import eht_lattice_blocks, assemble_k_matrices
            cell = validate_cell(torch.tensor(getattr(atoms, 'cell').array), pbc_flags)
            if self._pbc_mode == "eht-k":
                K, W = validate_kpoints(self._kpoints, self._kpoint_weights)
            else:  # 'eht-gamma'
                K = torch.zeros((1, 3), dtype=positions.dtype, device=self.device)
                W = torch.ones((1,), dtype=positions.dtype, device=self.device)
            # Basis setup
            basis = build_atom_basis(numbers, self._p_basis)
            # Build lattice blocks for EHT up to cutoffs
            blocks = eht_lattice_blocks(
                numbers,
                positions,
                basis,
                self._p_gxtb,
                self._schema,
                cell,
                float(self._pbc_cutoff),
                float(self._pbc_cn_cutoff),
            )
            mats = assemble_k_matrices(blocks['translations'], blocks['S_blocks'], blocks['H_blocks'], K)
            # Solve generalized eigenproblems per k using symmetric orthogonalization
            # and accumulate band energy (closed-shell, insulating filling with N_occ bands).
            Sks = mats['S_k']; Hks = mats['H_k']
            # Electron count heuristic consistent with non-periodic path (valence electrons present in basis)
            elem_to_shells: Dict[int, set[str]] = {}
            for sh in basis.shells:
                elem_to_shells.setdefault(int(sh.element), set()).add(sh.l)
            nelec = 0
            for z in numbers.tolist():
                val = _electron_configuration_valence_counts(int(z))
                present = elem_to_shells.get(int(z), set())
                nelec += int(round(sum(v for l, v in val.items() if l in present)))
            nocc = nelec // 2
            if nocc <= 0:
                raise ValueError("Computed zero occupied bands; check basis/element configuration.")
            E_band = torch.tensor(0.0, dtype=positions.dtype, device=self.device)
            for Sk, Hk, wk in zip(Sks, Hks, W):
                # Eigensolve S^{-1/2} H S^{-1/2}
                evals_S, U = torch.linalg.eigh(Sk)
                if torch.any(evals_S.real <= 0):
                    me = float(evals_S.real.min().item())
                    raise ValueError(f"S(k) not SPD (min eigenvalue {me:.3e}); reduce pbc_cutoff or review basis normalization.")
                X = (U * evals_S.clamp_min(1e-30).rsqrt()) @ U.conj().T
                H_ortho = X.conj().T @ Hk @ X
                evals, _ = torch.linalg.eigh(H_ortho)
                # Sum lowest nocc eigenvalues (real parts)
                E_band = E_band + wk.to(dtype=E_band.dtype) * evals.real[:nocc].sum()
            # Only electronic band energy is provided under PBC in this mode.
            self.results["energy"] = float(E_band.item() * EH2EV)
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

        # EEQBC charges: for now, require user-provided charges via atoms.info["q_eeqbc"] or raise
        # This avoids placeholders and keeps theory-consistent dependency.
        info = getattr(atoms, "info", {}) or {}
        q = info.get("q_eeqbc", None)
        if q is None:
            # Try tad_multicharge if available
            try:
                from tad_multicharge import get_eeq_charges  # type: ignore

                charges_np = get_eeq_charges(
                    numbers.cpu().numpy(), positions.cpu().numpy(), 0
                )
                q = charges_np
            except Exception as exc:  # pragma: no cover
                raise ValueError(
                    "EEQBC charges missing and tad_multicharge not available. "
                    "Provide atoms.info['q_eeqbc'] as (nat,) array to evaluate E_rep."
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
        # Enable shell-resolved second-order TB (doc/theory/15, Eqs. 98–106):
        # Build ShellSecondOrderParams from element Hubbard gamma as U^{(2),0} baseline and provide CN.
        so_params = None
        if self._enable_second_order:
            try:
                from .hamiltonian.second_order_tb import build_shell_second_order_params as _build_sp
                shell_params = _build_sp(int(numbers.max().item()), hub['gamma'])
                # Reuse CN computed above for current geometry
                so_params = {  # type: ignore[assignment]
                    'shell_params': shell_params,
                    'cn': cn,
                }
            except Exception as exc:
                # If shell second-order setup fails, we fall back to atomic second-order below
                so_params = None
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
        try:
            res = scf(
                numbers,
                positions,
                basis,
                builder,
                core['S'],
                hubbard=hub,
                ao_atoms=ao_atoms,
                nelec=nelec,
                max_iter=50,
                tol=1e-6,
                second_order=self._enable_second_order,
                so_params=so_params if so_params is not None else {},
                eeq_charges=charges,
                dynamic_overlap=bool(qv_pack is not None),
                qvszp_params=qv_pack if qv_pack is not None else None,
                aes=self._enable_aes and (aes_params_dict is not None),
                aes_params=aes_params_dict,
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
            res = scf(
                numbers,
                positions,
                basis,
                builder,
                core['S'],
                hubbard=hub,
                ao_atoms=ao_atoms,
                nelec=nelec,
                max_iter=50,
                tol=1e-6,
                second_order=self._enable_second_order,
                so_params=so_atomic,
                eeq_charges=charges,
                dynamic_overlap=bool(qv_pack is not None),
                qvszp_params=qv_pack if qv_pack is not None else None,
                aes=self._enable_aes and (aes_params_dict is not None),
                aes_params=aes_params_dict,
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
        self.results["energy"] = float(e_total.item() * EH2EV)

    def get_forces(self, atoms):  # type: ignore
        # Numerical forces via central differences on energy in eV
        eps = 1.0e-3  # Angstrom
        f = []
        pos = atoms.get_positions()
        for a in range(len(atoms)):
            f_comp = [0.0, 0.0, 0.0]
            for k in range(3):
                dp = pos.copy()
                dp[a, k] += eps
                atoms.set_positions(dp)
                ep = self.get_potential_energy(atoms)
                dm = pos.copy()
                dm[a, k] -= eps
                atoms.set_positions(dm)
                em = self.get_potential_energy(atoms)
                # Force component in eV/Angstrom
                f_comp[k] = - (ep - em) / (2 * eps)
            f.append(f_comp)
        # Restore original positions
        atoms.set_positions(pos)
        return torch.tensor(f, dtype=torch.get_default_dtype()).cpu().numpy()
