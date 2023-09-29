from typing import List

import numpy as np

import torch

from pcs.layers import MonotonicBinaryEmbeddings, BornBinaryEmbeddings
from pcs.layers.candecomp import MonotonicCPLayer, BornCPLayer
from pcs.models import MonotonicPC, BornPC, PC
from pcs.sampling import inverse_transform_sample
from region_graph.random_binary_tree import RandomBinaryTree
from tests.test_likelihood import check_mar_ll_pf
from tests.test_utils import generate_all_binary_samples


def check_sampling(
        model: PC,
        num_samples: int
):
    samples = inverse_transform_sample(
        model,
        vdomain=2,
        num_samples=num_samples
    )
    num_variables = model.num_variables
    assert samples.shape == (num_samples, num_variables)
    assert torch.all(torch.isin(samples, torch.tensor([0, 1])))


def check_conditional_sampling(
        model: PC,
        num_samples: int,
        evi_vars: List[int],
        evi_state: torch.Tensor,
        mis_var: int
):
    num_variables = model.num_variables
    samples = inverse_transform_sample(
        model,
        vdomain=2,
        num_samples=num_samples,
        evidence=(evi_vars, evi_state)
    )
    assert samples.shape == (num_samples, num_variables)
    assert torch.all(samples[:, evi_vars] == evi_state)
    assert torch.all(torch.isin(samples[:, mis_var], torch.tensor([0, 1])))

    cond_samples = torch.zeros(2, num_variables, dtype=torch.long)
    cond_samples[0, evi_vars] = evi_state
    cond_samples[1, evi_vars] = evi_state
    cond_samples[0, mis_var] = 0
    cond_samples[1, mis_var] = 1
    evi_log_probs = model.log_prob(cond_samples)
    mar_mask = torch.zeros(2, num_variables, dtype=torch.bool)
    mar_mask[:, mis_var] = True
    mar_log_probs = model.log_marginal_prob(cond_samples, mar_mask)
    con_log_probs = evi_log_probs - mar_log_probs
    con_log_probs = con_log_probs.squeeze(dim=1)
    assert np.isclose(torch.logsumexp(con_log_probs, dim=0).item(), 0.0)

    emprical_freqs = torch.zeros(2, dtype=torch.long)
    emprical_freqs[1] = torch.sum(samples[:, mis_var])
    emprical_freqs[0] = samples.shape[0] - emprical_freqs[1]
    emprical_freqs = torch.log(emprical_freqs) - np.log(samples.shape[0])
    assert torch.allclose(con_log_probs, emprical_freqs, atol=2.5e-3)  # 99.5% confidence


def test_sampling_monotonic_pc():
    num_replicas = 2
    depth = 2
    num_components = 3
    num_variables = 5
    num_samples, num_con_samples = 10000, 600000
    mis_var, evi_vars, evi_state = 1, [0, 2, 3, 4], torch.tensor([1, 0, 0, 1])

    rg = RandomBinaryTree(num_variables, num_repetitions=num_replicas, depth=depth)
    model = MonotonicPC(
        rg, input_layer_cls=MonotonicBinaryEmbeddings, compute_layer_cls=MonotonicCPLayer,
        num_components=num_components)
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_mar_ll_pf(model, data)
    check_sampling(model, num_samples)
    check_conditional_sampling(model, num_con_samples, evi_vars, evi_state, mis_var)


def test_sampling_born_pc():
    num_replicas = 2
    depth = 2
    num_components = 3
    num_variables = 5
    num_samples, num_con_samples = 10000, 600000
    mis_var, evi_vars, evi_state = 1, [0, 2, 3, 4], torch.tensor([1, 0, 0, 1])

    rg = RandomBinaryTree(num_variables, num_repetitions=num_replicas, depth=depth)
    model = BornPC(
        rg, input_layer_cls=BornBinaryEmbeddings, compute_layer_cls=BornCPLayer,
        num_components=num_components)
    data = torch.LongTensor(generate_all_binary_samples(num_variables))
    check_mar_ll_pf(model, data)
    check_sampling(model, num_samples)
    check_conditional_sampling(model, num_con_samples, evi_vars, evi_state, mis_var)
