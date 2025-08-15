# 🚧 g‑xTB (Torch) 🚧

This repository is a Python implementation of g‑xTB accelerated by PyTorch. The theory and equations are mapped directly from the ChemRxiv preprint:

- https://chemrxiv.org/engage/chemrxiv/article-details/685434533ba0887c335fc974

This is a preliminary version of g‑xTB, a general‑purpose semiempirical quantum mechanical method approximating ωB97M‑V/def2‑TZVPPD properties.

For an ASE interface see `src/gxtb/ase_calc.py` (`GxTBCalculator`).


## Implementation Status

| Area | Theory ref | Module(s) | Status | Notes |
|---|---|---|---|---|
| Parameter loaders | My GUESS WORK | `params/loader.py`, `params/types.py`, `params/schema.py` | Partial | Its based on my observation and guesses |
| Coordination numbers (molecules) | `doc/theory/9_cn.md` | `cn.py` | Done | Smooth CN per Eq. 47 with Torch vectorization. |
| q‑vSZP basis (static + dynamic) | `doc/theory/7_q-vSZP_basis_set.md` | `basis/qvszp.py` | Partial | Dynamic coeffs via q_eff wired; full tests pending. |
| Overlap + diatomic scaling | `doc/theory/8_diatomic_frame_scaled_overlap.md` | `basis/overlap.py`, `basis/md_overlap.py`, `hamiltonian/overlap_tb.py` | Done | σ/π/δ scaling up to f; analytic blocks. |
| EHT Hamiltonian + Wolfsberg | `doc/theory/12_eht_hamiltonian.md` | `hamiltonian/eht.py`, `hamiltonian/distance_tb.py`, `hamiltonian/onsite_tb.py` | Partial | Core done; EN penalty and distance polynomials present; CN‑onsite linear model. |
| First‑order TB (E^(1)) | `doc/theory/14_first_order_tb.md` | `hamiltonian/first_order.py` | Partial | Structure present; switching functions and full parameter mapping in progress. |
| Second‑order TB (isotropic) | `doc/theory/15_second_order_tb.md` | `hamiltonian/second_order_tb.py`, `scf.py` | Partial | Atomic + shell paths; shell reference populations and CN used; tests expanding. |
| Anisotropic electrostatics (AES) | `doc/theory/16_anisotropic_electrostatics.md` | `hamiltonian/aes.py`, `hamiltonian/moments_builder.py` | Partial | Multipole moments and damping implemented; integration paths guarded and optional. |
| Spin polarization | `doc/theory/17_spin_polarization.md` | `hamiltonian/spin.py`, `scf.py` | Done | UHF spin energy and Fock add‑on; shell magnetizations. |
| Third‑order TB (E^(3)) | `doc/theory/18_third_order_tb.md` | `hamiltonian/third_order.py`, `scf.py` | Partial | Parameter mapping and tau3 matrix present; validation ongoing. |
| Fourth‑order TB (E^(4)) | `doc/theory/19_fourth_order_tb.md` | `hamiltonian/fourth_order.py`, `scf.py` | Partial | Onsite Fock term and energy wired; requires tuned γ4. |
| MFX exchange (long‑range) | `doc/theory/20_mfx.md` | `hamiltonian/mfx.py`, `scf.py` | Partial | γ^MFX AO build and hooks present; screening/validation pending. |
| OFX exchange (onsite) | `doc/theory/21_ofx.md` | `hamiltonian/ofx.py`, `scf.py` | Done | Energy and Fock per Eqs. 155/159; Λ^0 explicit, no defaults. |
| Atomic increments | `doc/theory/10_atomic_energy_increment.md` | `classical/increment.py` | Done | Element‑resolved constants loaded via schema. |
| Semi‑classical repulsion | `doc/theory/11_semi_classical_repulsion.md` | `classical/repulsion.py` | Partial | Kernel implemented with CN/charge coupling; tuning/validation pending. |
| Dispersion (revD4) | `doc/theory/22_dft_revd4.md` | `classical/dispersion.py` | Partial | Method/TOML loaders and energy path; SCF coupling off by design. |
| SCF solver | `doc/theory/5_kohn_sham_type_equations.md` | `scf.py`, `hamiltonian/scf_adapter.py` | Partial | Löwdin orthogonalization, Mulliken, linear/Anderson/Broyden mixing, dynamic overlap; convergence heuristics in place. |
| Nuclear gradients | `doc/theory/6_nuclear_gradients.md` | `grad/nuclear.py` | Todo | Analytic gradients pending for all terms. |

### Periodic Boundary Conditions (PBC)
| PBC Area | Theory ref | Module(s) | Status | Notes |
|---|---|---|---|---|
| Cell/k‑point utilities | `doc/theory/25_periodic_boundary_conditions.md` | `pbc/cell.py`, `pbc/kpoints.py` | Done | Cell validation, real‑space cutoffs, Monkhorst–Pack grids. |
| EHT lattice blocks | `doc/theory/25` + `doc/theory/12` | `pbc/bloch.py` | Partial | S(0R), H(0R) via diatomic overlaps and CN‑onsite; real‑space cutoffs. |
| Bloch sums S(k), H(k) | `doc/theory/25` | `pbc/bloch.py` | Done | Hermitian assembly and Γ/general k support. |
| Band energy (Γ/k) | `doc/theory/25` | `ase_calc.py` | Done | Symmetric orthogonalization per k; insulating filling heuristic. |
| PBC SCF (k‑resolved) | `doc/theory/25` | — | Todo | Requires k‑resolved density build and Ewald‑type second‑order. |
| PBC second‑order/AES | `doc/theory/25` | — | Todo | Not yet supported under PBC; guarded by explicit errors. |
| PBC dispersion (D4) | `doc/theory/25` + `doc/theory/22` | — | Todo | Periodic D4 pending; explicit error in ASE path. |
| PBC forces/stress | `doc/theory/6` + `doc/theory/25` | — | Todo | Gradient/stress machinery pending. |
