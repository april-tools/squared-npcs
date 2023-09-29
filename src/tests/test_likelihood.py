import itertools
from typing import Tuple, Optional

import numpy as np
import pytest
import torch
from scipy import integrate

from pcs.hmm import MonotonicHMM, BornHMM
from pcs.layers.mixture import MonotonicMixtureLayer, BornMixtureLayer
from pcs.layers.tucker import MonotonicTucker2Layer, BornTucker2Layer
from pcs.layers.candecomp import MonotonicCPLayer, BornCPLayer
from pcs.layers.input import MonotonicBinaryEmbeddings, BornBinaryEmbeddings, NormalDistribution, \
    MultivariateNormalDistribution, BornNormalDistribution, BornMultivariateNormalDistribution, MonotonicBSplines, \
    BornBSplines, MonotonicBinomial, BornBinomial, MonotonicEmbeddings, BornEmbeddings
from pcs.models import MonotonicPC, BornPC, PC
from region_graph import RegionGraph, RegionNode
from region_graph.linear_vtree import LinearVTree
from region_graph.quad_tree import QuadTree
from region_graph.random_binary_tree import RandomBinaryTree

from tests.test_utils import generate_all_binary_samples, generate_all_ternary_samples


def check_evi_ll(model: PC, x: torch.Tensor) -> torch.Tensor:
    lls = model.log_prob(x)
    assert lls.shape == (len(x), 1)
    assert torch.all(torch.isfinite(lls))
    assert torch.allclose(torch.logsumexp(lls, dim=0).exp(), torch.tensor(1.0), atol=1e-15)
    return lls


def check_mar_ll_pf(model: PC, x: torch.Tensor):
    mar_mask = torch.ones_like(x, dtype=torch.bool)
    lls = model.log_marginal_score(x, mar_mask)
    log_z = model.log_pf()
    assert torch.allclose(lls, log_z, atol=1e-15)


def check_mar_ll_one(model: PC, x: torch.Tensor, num_mar_variables: int = 1, arity: int = 2) -> torch.Tensor:
    assert x.shape[1] > num_mar_variables
    num_mar_samples = arity ** num_mar_variables
    z = x[:num_mar_samples]
    mar_mask = torch.zeros_like(z, dtype=torch.bool)
    mar_mask[:, :z.shape[1] - num_mar_variables] = True
    lls = model.log_marginal_prob(z, mar_mask)
    assert torch.allclose(torch.logsumexp(lls, dim=0).exp(), torch.tensor(1.0), atol=1e-15)
    return lls


def check_pdf(model, interval: Optional[Tuple[float, float]] = None):
    pdf = lambda y, x: torch.exp(model.log_prob(torch.Tensor([[x, y]])))
    if interval is None:
        a, b = -np.inf, np.inf
    else:
        a, b = interval
    ig, err = integrate.dblquad(pdf, a, b, a, b)
    assert np.isclose(ig, 1.0, atol=1e-15)


@pytest.mark.parametrize("compute_layer,num_variables,num_replicas,depth,num_components,input_mixture",
                         list(itertools.product(
                             [MonotonicCPLayer],
                             [8, 13], [1, 4], [-1, 1, 3], [1, 3], [False, True]
                         )))
def test_monotonic_pc_random(compute_layer, num_variables, num_replicas, depth, num_components, input_mixture):
    rg = RandomBinaryTree(num_variables, num_repetitions=num_replicas, depth=depth)
    model = MonotonicPC(
        rg, input_layer_cls=MonotonicBinaryEmbeddings, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components)
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 5, num_variables - 1]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables)


@pytest.mark.parametrize("compute_layer,num_variables,num_replicas,depth,num_components,input_mixture,exp_reparam",
                         list(itertools.product(
                             [BornCPLayer],
                             [8, 13], [1, 4], [-1, 1, 3], [1, 3], [False, True], [False, True]
                         )))
def test_born_pc_random(compute_layer, num_variables, num_replicas, depth, num_components, input_mixture, exp_reparam):
    rg = RandomBinaryTree(num_variables, num_repetitions=num_replicas, depth=depth)
    init_method = 'log-normal' if exp_reparam else 'uniform'
    compute_layer_kwargs = input_layer_kwargs = {'exp_reparam': exp_reparam, 'init_method': init_method}
    model = BornPC(
        rg, input_layer_cls=BornBinaryEmbeddings, compute_layer_cls=compute_layer,
        input_layer_kwargs=input_layer_kwargs, compute_layer_kwargs=compute_layer_kwargs,
        input_mixture=input_mixture, num_components=num_components)
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 5, num_variables - 1]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables)


@pytest.mark.parametrize("compute_layer,image_shape,num_components,input_mixture",
                         list(itertools.product(
                             [MonotonicCPLayer],
                             [(1, 3, 3)], [1, 3], [False, True]
                         )))
def test_monotonic_pc_pseudo_small_image(compute_layer, image_shape, num_components, input_mixture):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = MonotonicPC(
        rg, input_layer_cls=MonotonicBinaryEmbeddings, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components)
    data = torch.LongTensor(generate_all_binary_samples(np.prod(image_shape).item()))
    check_evi_ll(model, data)


@pytest.mark.parametrize("compute_layer,image_shape,num_components,input_mixture",
                         list(itertools.product(
                             [BornCPLayer],
                             [(1, 3, 3)], [1, 3], [False, True]
                         )))
def test_born_pc_pseudo_small_image(compute_layer, image_shape, num_components, input_mixture):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = BornPC(
        rg, input_layer_cls=BornBinaryEmbeddings, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components)
    data = torch.LongTensor(generate_all_binary_samples(np.prod(image_shape).item()))
    check_evi_ll(model, data)


@pytest.mark.parametrize("compute_layer,image_shape,num_components,input_mixture",
                         list(itertools.product(
                             [MonotonicCPLayer],
                             [(1, 7, 7), (3, 28, 28)], [1, 3], [False, True]
                         )))
def test_monotonic_pc_pseudo_large_image(compute_layer, image_shape, num_components, input_mixture):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = MonotonicPC(
        rg, input_layer_cls=MonotonicEmbeddings, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components,
        input_layer_kwargs={'num_states': 768})
    data = torch.round(torch.rand((42, np.prod(image_shape)))).long()
    lls = model.log_prob(data)
    assert lls.shape == (len(data), 1)


@pytest.mark.parametrize("compute_layer,image_shape,num_components,input_mixture",
                         list(itertools.product(
                             [BornCPLayer],
                             [(1, 7, 7), (3, 28, 28)], [1, 3], [False, True]
                         )))
def test_born_pc_pseudo_large_image(compute_layer, image_shape, num_components, input_mixture):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = BornPC(
        rg, input_layer_cls=BornEmbeddings, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components,
        input_layer_kwargs={'num_states': 768})
    data = torch.round(torch.rand((42, np.prod(image_shape)))).long()
    lls = model.log_prob(data)
    assert lls.shape == (len(data), 1)


@pytest.mark.parametrize("compute_layer,image_shape,num_components,input_mixture",
                         list(itertools.product(
                             [MonotonicCPLayer],
                             [(1, 7, 7)], [1, 3], [False, True]
                         )))
def test_monotonic_pc_image_dequantize(compute_layer, image_shape, num_components, input_mixture):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = MonotonicPC(
        rg, input_layer_cls=NormalDistribution, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components, dequantize=True)
    data = torch.round(torch.rand((42, np.prod(image_shape)))).long()
    data = (data + torch.rand(*data.shape)) / 2.0
    logit_data, ldj = model._logit(data)
    unlogit_data, ildj = model._unlogit(logit_data)
    assert torch.allclose(data, unlogit_data)
    assert torch.allclose(ldj + ildj, torch.zeros(()))
    lls = model.log_prob(data)
    assert lls.shape == (len(data), 1)


@pytest.mark.parametrize("compute_layer,image_shape,num_components,input_mixture",
                         list(itertools.product(
                             [BornCPLayer],
                             [(1, 7, 7)], [1, 3], [False, True]
                         )))
def test_born_pc_image_dequantize(compute_layer, image_shape, num_components, input_mixture):
    rg = QuadTree(image_shape, struct_decomp=True)
    model = BornPC(
        rg, input_layer_cls=BornNormalDistribution, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components, dequantize=True)
    data = torch.round(torch.rand((42, np.prod(image_shape)))).long()
    data = (data + torch.rand(*data.shape)) / 2.0
    logit_data, ldj = model._logit(data)
    unlogit_data, ildj = model._unlogit(logit_data)
    assert torch.allclose(data, unlogit_data)
    assert torch.allclose(ldj + ildj, torch.zeros(()))
    lls = model.log_prob(data)
    assert lls.shape == (len(data), 1)


@pytest.mark.parametrize("compute_layer,num_variables,num_components,input_mixture,num_replicas",
                         list(itertools.product(
                             [MonotonicCPLayer, MonotonicTucker2Layer],
                             [8, 13], [1, 3], [False, True], [1, 4]
                         )))
def test_monotonic_pc_linear_rg(compute_layer, num_variables, num_components, input_mixture, num_replicas):
    rg = LinearVTree(num_variables, num_repetitions=num_replicas, randomize=True)
    model = MonotonicPC(
        rg, input_layer_cls=MonotonicBinaryEmbeddings, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components)
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 5, num_variables - 1]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables)


@pytest.mark.parametrize("compute_layer,num_variables,num_components,input_mixture,num_replicas",
                         list(itertools.product(
                             [BornCPLayer, BornTucker2Layer],
                             [8, 13], [1, 3], [False, True], [1, 4]

                         )))
def test_born_pc_linear_rg(compute_layer, num_variables, num_components, input_mixture, num_replicas):
    rg = LinearVTree(num_variables, num_repetitions=num_replicas, randomize=True)
    model = BornPC(
        rg, input_layer_cls=BornBinaryEmbeddings, compute_layer_cls=compute_layer,
        input_mixture=input_mixture, num_components=num_components)
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 5, num_variables - 1]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables)


@pytest.mark.parametrize("compute_layer,num_variables,depth,num_components",
                         list(itertools.product(
                             [MonotonicCPLayer],
                             [4, 7], [1, 2], [1, 3]
                         )))
def test_monotonic_binomial_pc(compute_layer, num_variables, depth, num_components):
    rg = RandomBinaryTree(num_variables, num_repetitions=1, depth=depth)
    model = MonotonicPC(
        rg, input_layer_cls=MonotonicBinomial, compute_layer_cls=compute_layer,
        num_components=num_components, input_layer_kwargs={'num_states': 3})
    data = torch.LongTensor(generate_all_ternary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 3]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables, arity=3)


@pytest.mark.parametrize("compute_layer,num_variables,depth,num_components",
                         list(itertools.product(
                             [BornCPLayer],
                             [4, 7], [1, 2], [1, 3]
                         )))
def test_born_binomial_pc(compute_layer, num_variables, depth, num_components):
    rg = RandomBinaryTree(num_variables, num_repetitions=1, depth=depth)
    model = BornPC(
        rg, input_layer_cls=BornBinomial, compute_layer_cls=compute_layer,
        num_components=num_components, input_layer_kwargs={'num_states': 3})
    data = torch.LongTensor(generate_all_ternary_samples(num_variables))
    check_evi_ll(model, data)
    check_mar_ll_pf(model, data)
    for num_mar_variables in [1, 3]:
        check_mar_ll_one(model, data, num_mar_variables=num_mar_variables, arity=3)


@pytest.mark.parametrize("compute_layer,num_components",
                         list(itertools.product([MonotonicCPLayer], [3])))
def test_normal_monotonic_pc(compute_layer, num_components):
    rg = RandomBinaryTree(2, num_repetitions=1, depth=1)
    model = MonotonicPC(
        rg, input_layer_cls=NormalDistribution, compute_layer_cls=compute_layer,
        num_components=num_components)
    model.eval()
    check_pdf(model)


def test_multivariate_monotonic_normal_pc():
    rg = RegionGraph()
    rg.add_node(RegionNode([0, 1]))
    model = MonotonicPC(
        rg, input_layer_cls=MultivariateNormalDistribution, out_mixture_layer_cls=MonotonicMixtureLayer,
        num_components=3)
    model.eval()
    check_pdf(model)


def test_normal_born_pc():
    rg = RandomBinaryTree(2, num_repetitions=1, depth=1)
    model = BornPC(
        rg, input_layer_cls=BornNormalDistribution,
        out_mixture_layer_cls=BornMixtureLayer,
        num_components=3)
    model.eval()
    check_pdf(model)


def test_multivariate_normal_born_pc():
    rg = RegionGraph()
    rg.add_node(RegionNode([0, 1]))
    model = BornPC(
        rg, input_layer_cls=BornMultivariateNormalDistribution, out_mixture_layer_cls=BornMixtureLayer,
        num_components=3)
    model.eval()
    check_pdf(model)


@pytest.mark.parametrize("compute_layer,num_components",
                         list(itertools.product([MonotonicCPLayer], [2])))
def test_spline_monotonic_pc(compute_layer, num_components):
    rg = RandomBinaryTree(2, num_repetitions=1, depth=1)
    model = MonotonicPC(
        rg, input_layer_cls=MonotonicBSplines, compute_layer_cls=compute_layer,
        num_components=num_components, input_layer_kwargs={'order': 2, 'num_knots': 6, 'interval': (0.0, 1.0)})
    model.eval()
    check_pdf(model, interval=(0.0, 1.0))


@pytest.mark.parametrize("compute_layer,num_components,exp_reparam",
                         list(itertools.product([BornCPLayer], [2], [False, True])))
def test_spline_born_pc(compute_layer, num_components, exp_reparam):
    rg = RandomBinaryTree(2, num_repetitions=1, depth=1)
    init_method = 'log-normal' if exp_reparam else 'uniform'
    model = BornPC(
        rg, input_layer_cls=BornBSplines, compute_layer_cls=compute_layer,
        num_components=num_components,
        input_layer_kwargs={'order': 2, 'num_knots': 6, 'interval': (0.0, 1.0), 'init_method': init_method},
        compute_layer_kwargs={'init_method': init_method, 'exp_reparam': exp_reparam}
    )
    model.eval()
    check_pdf(model, interval=(0.0, 1.0))


@pytest.mark.parametrize("seq_length,hidden_size",
                         list(itertools.product([2, 7], [1, 13])))
def test_monotonic_hmm(seq_length, hidden_size):
    model = MonotonicHMM(vocab_size=3, seq_length=seq_length, hidden_size=hidden_size)
    data = torch.LongTensor(generate_all_ternary_samples(seq_length))
    check_evi_ll(model, data)


@pytest.mark.parametrize("seq_length,hidden_size",
                         list(itertools.product([2, 7], [1, 13])))
def test_born_hmm(seq_length, hidden_size):
    model = BornHMM(vocab_size=3, seq_length=seq_length, hidden_size=hidden_size)
    data = torch.LongTensor(generate_all_ternary_samples(seq_length))
    check_evi_ll(model, data)
