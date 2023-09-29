import itertools
import pytest
import numpy as np

from pcs.utils import log_binomial


def generate_all_nary_samples(num_variables: int, arity: int = 2) -> np.ndarray:
    vs = list(range(arity))
    return np.asarray(list(itertools.product(vs, repeat=num_variables)))


def generate_all_binary_samples(num_variables: int) -> np.ndarray:
    return generate_all_nary_samples(num_variables, arity=2)


def generate_all_ternary_samples(num_variables: int) -> np.ndarray:
    return generate_all_nary_samples(num_variables, arity=3)


@pytest.mark.parametrize("num_variables,arity", list(itertools.product([1, 5], [2, 3])))
def test_generate_all_nary_samples(num_variables, arity):
    x = generate_all_nary_samples(num_variables, arity=arity)
    assert len(x) == int(arity ** num_variables)
    assert np.all(np.isin(x, list(range(arity))))


@pytest.mark.parametrize("n,k", [(3, 0), (3, 1), (3, 2), (5, 2)])
def test_log_binomial(n, k):
    def factorial(x):
        return np.prod(1 + np.asarray(range(x)))
    lb = log_binomial(n, k)
    lb_true = np.log(factorial(n) / (factorial(k) * factorial(n - k)))
    assert np.isclose(lb, lb_true)
