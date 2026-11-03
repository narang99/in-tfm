import numpy as np

from in_tfm.attribution.shape_shapley import exact_shapley, shape_shapley


def test_shares_add_up_to_one():
    parts = np.random.default_rng(0).normal(size=(12, 8))
    shares = shape_shapley(parts, top=4)
    assert np.isclose(shares.shares.sum() + shares.rest, 1.0)


def test_copies_split_the_credit_equally():
    part = np.random.default_rng(1).normal(size=8)
    other = np.random.default_rng(2).normal(size=8) * 0.1
    shares = exact_shapley(np.stack([part, part, other]), 2 * part + other)
    assert np.isclose(shares[0], shares[1])
    assert shares[0] > shares[2]


def test_a_zero_source_gets_nothing():
    parts = np.random.default_rng(3).normal(size=(3, 8))
    parts[1] = 0.0
    assert np.isclose(exact_shapley(parts, parts.sum(axis=0))[1], 0.0)


def test_one_source_gets_everything():
    part = np.random.default_rng(4).normal(size=(1, 8))
    assert np.isclose(exact_shapley(part, part[0])[0], 1.0)
