import torch
import pytest

from gxtb.scf import scf


def _minimal_system():
    numbers = torch.tensor([1], dtype=torch.long)
    positions = torch.zeros((1,3), dtype=torch.float64)
    # Minimal 1s AO basis
    from types import SimpleNamespace
    shells = [SimpleNamespace(atom_index=0, element=1, l='s', nprims=1, primitives=[(1.0, 1.0, 0.0)])]
    ao_counts = [1]; ao_offsets = [0]
    basis = SimpleNamespace(shells=shells, ao_counts=ao_counts, ao_offsets=ao_offsets, nao=1)
    # Overlap S=1, core builder zeros, ao_atoms=[0]
    S = torch.ones((1,1), dtype=torch.float64)
    ao_atoms = torch.tensor([0], dtype=torch.long)
    def builder(numbers, positions, ctx):
        return { 'H0': torch.zeros((1,1), dtype=torch.float64), 'S': S, 'ao_atoms': ao_atoms }
    hub = {'gamma': torch.zeros(10, dtype=torch.float64)}
    return numbers, positions, basis, builder, S, hub, ao_atoms


def test_scf_aes_missing_params_raises():
    numbers, positions, basis, builder, S, hub, ao_atoms = _minimal_system()
    with pytest.raises(ValueError):
        scf(numbers, positions, basis, builder, S, hub, ao_atoms, nelec=0, aes=True, aes_params=None)


def test_scf_dispersion_missing_params_raises():
    numbers, positions, basis, builder, S, hub, ao_atoms = _minimal_system()
    with pytest.raises(ValueError):
        scf(numbers, positions, basis, builder, S, hub, ao_atoms, nelec=0, dispersion=True, dispersion_params=None)

