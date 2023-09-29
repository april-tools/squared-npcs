from typing import Optional, Tuple, List, Union

import torch

from pcs.models import PC


@torch.no_grad()
def inverse_transform_sample(
        model: PC,
        *,
        vdomain: int,
        num_samples: int = 1,
        evidence: Optional[Tuple[List[int], torch.Tensor]] = None,
        device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    num_variables = model.num_variables
    mar_mask = torch.ones(num_samples, vdomain, num_variables, dtype=torch.bool, device=device)
    mar_data = torch.zeros(num_samples, vdomain, num_variables, dtype=torch.long, device=device)
    samples = torch.zeros(num_samples, num_variables, dtype=torch.long, device=device)
    variables_to_sample = range(num_variables)

    if evidence is not None:
        evidence_variables, evidence_state = evidence
        mar_mask[:, :, evidence_variables] = False
        mar_data[:, :, evidence_variables] = evidence_state
        variables_to_sample = filter(lambda v: v not in evidence_variables, variables_to_sample)
        samples[:, evidence_variables] = evidence_state
        samples_mar_mask = torch.ones(1, num_variables, dtype=torch.long, device=device)
        samples_mar_mask[:, evidence_variables] = False
        mar_log_probs = model.log_marginal_prob(samples[[0]], samples_mar_mask)
        mar_log_probs = mar_log_probs.squeeze(dim=0).unsqueeze(dim=1)
    else:
        mar_log_probs = torch.zeros(num_samples, 1, device=device)

    for i in variables_to_sample:
        mar_mask[:, :, i] = False
        mar_data[:, :, i] = torch.arange(vdomain, device=device).unsqueeze(dim=0)

        flat_mar_data = mar_data.view(-1, num_variables)
        flat_mar_mask = mar_mask.view(-1, num_variables)
        log_probs = model.log_marginal_prob(flat_mar_data, flat_mar_mask)
        # log p(X_1 = a_1, ..., X_{i-1} = a_{i-1}, X_i = v_j)
        log_probs = log_probs.view(num_samples, vdomain)
        # p(X_i = v_j | X_1 = a_1, ..., X_{i-1} = a_{i-1})
        con_probs = torch.exp(log_probs - mar_log_probs)

        u = torch.rand(con_probs.shape[0], 1, device=device)
        sampled_mask = u <= torch.cumsum(con_probs, dim=1)
        sampled_value = vdomain - sampled_mask.long().sum(dim=1)  # (num_samples,)
        samples[:, i] = sampled_value

        idx_range = torch.arange(num_samples, device=device)
        mar_log_probs = log_probs[idx_range, sampled_value].unsqueeze(dim=1)
        mar_data[idx_range, :, i] = sampled_value.unsqueeze(dim=1)

    return samples
