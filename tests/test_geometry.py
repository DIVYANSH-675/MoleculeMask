import torch

from moleculemask.model import MaskedMolecule, one_mask, synthetic


def fixture():
    p, z, _ = synthetic(4, seed=1)
    return torch.tensor(p), torch.tensor(z), one_mask(4, 6, False)


def test_rigid_motion_equivariance_and_invariant_property():
    torch.manual_seed(7)
    model = MaskedMolecule().double()
    p, z, mask = fixture()
    p = p.double()
    mask = mask.double()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    shift = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
    x, prop = model(p, z, mask)
    transformed, prop2 = model(p @ q + shift, z, mask)
    torch.testing.assert_close(transformed, x @ q + shift)
    torch.testing.assert_close(prop, prop2)


def test_hidden_coordinates_cannot_leak():
    p, z, mask = fixture()
    model = MaskedMolecule()
    first = model(p, z, mask)
    second = model(p + 100 * mask[..., None], z, mask)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_atom_permutation_equivariance():
    p, z, mask = fixture()
    model = MaskedMolecule()
    order = torch.tensor([3, 1, 5, 0, 2, 4])
    pred, prop = model(p, z, mask)
    pred2, prop2 = model(p[:, order], z[:, order], mask[:, order])
    torch.testing.assert_close(pred2, pred[:, order])
    torch.testing.assert_close(prop2, prop)


def test_single_molecule_backward():
    p, z, mask = fixture()
    model = MaskedMolecule()
    pred, prop = model(p[:1], z[:1], mask[:1])
    loss = (pred - p[:1]).square().mean() + prop.square().mean()
    loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())


def test_loader_rejects_values_that_overflow_training_dtype(tmp_path):
    import numpy as np
    import pytest

    from moleculemask.model import load_data

    p, z, y = synthetic(5)
    for field in ("positions", "properties"):
        arrays = dict(
            positions=p.astype(np.float64), atomic_numbers=z, properties=y.astype(np.float64)
        )
        arrays[field].flat[0] = 1e100
        path = tmp_path / (field + ".npz")
        np.savez(path, **arrays)
        with pytest.raises(ValueError, match="finite float32"):
            load_data(path)
    np.savez(tmp_path / "valid.npz", positions=p, atomic_numbers=z, properties=y)
    loaded = load_data(tmp_path / "valid.npz")
    np.testing.assert_array_equal(loaded[0], p)


def test_direct_model_calls_validate_positions_and_elements():
    import pytest

    p, z, mask = fixture()
    model = MaskedMolecule()
    invalid = p.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Positions"):
        model(invalid, z, mask)
    with pytest.raises(ValueError, match="Atomic numbers"):
        model(p, z.float(), mask)
    invalid_z = z.clone()
    invalid_z[0, 0] = 119
    with pytest.raises(ValueError, match="Atomic numbers"):
        model(p, invalid_z, mask)


def test_boolean_masks_match_numeric_masks_without_mutating_inputs():
    p, z, mask = fixture()
    model = MaskedMolecule()
    boolean = mask.bool()
    expected = model(p, z, mask)
    actual = model(p, z, boolean)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert boolean.dtype == torch.bool
