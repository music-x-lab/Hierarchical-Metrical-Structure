import torch.nn as nn
import torch
import torch.nn.functional as F
from mir.nn.data_storage import FramedRAMDataStorage
from mir.nn.data_provider import DataProvider, default_collate, data_type_fix
from mir.nn.train import NetworkBehavior, NetworkInterface
from mir.nn.data_provider import FramedDataProvider, data_type_fix
from modules.vae import Reparameterizer
import numpy as np
from torch import jit, Tensor

@jit.script
def safe_logsumexp(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))

class CRFDecoder(nn.Module):

    def __init__(self, log_transition, transition_as_parameter=True):
        super().__init__()
        self.n_state = log_transition.shape[0]
        if (transition_as_parameter):
            self.log_transition = nn.Parameter(log_transition, requires_grad=False)
        else:
            self.log_transition = log_transition

    def viterbi_decode(self, log_observation):
        batch_dim, seq_length, n_state = log_observation.shape
        assert(n_state == self.n_state)
        with torch.no_grad():
            pre = torch.full((batch_dim, seq_length, n_state), -1, device=log_observation.device)
            forward_var = torch.zeros(batch_dim, self.n_state, device=log_observation.device)
            for i in range(seq_length):
                if (i == 0):
                    pre[:, i, :] = -1
                else:
                    forward_state_var = forward_var[:, :, None] + self.log_transition[None]
                    pre[:, i, :] = torch.argmax(forward_state_var, dim=1)
                    forward_var = torch.max(forward_state_var, dim=1).values
                forward_var += log_observation[:, i, :]
            result = torch.zeros((batch_dim, seq_length), device=log_observation.device, dtype=int)
            result[:, seq_length - 1] = torch.argmax(forward_var, dim=-1)
            for i in range(seq_length - 2, -1, -1):
                result[:, i] = pre[:, i + 1, result[:, i + 1]]
            return result

    def neg_log_likelihood(self, gt_seq, token_mask, log_observation):
        # score the observation on a partially observed hidden sequence
        seq_length, n_state = log_observation.shape
        assert(n_state == self.n_state)
        forward_var = torch.zeros(self.n_state, device=log_observation.device)
        forward_var_all = torch.zeros(self.n_state, device=log_observation.device)
        for i in range(0, seq_length):
            if (i > 0):
                forward_state_var = forward_var[:, None] + self.log_transition
                forward_var = safe_logsumexp(forward_state_var, dim=0) + log_observation[i, :]
                forward_state_var_all = forward_var_all[:, None] + self.log_transition
                forward_var_all = safe_logsumexp(forward_state_var_all, dim=0) + log_observation[i, :]
            else:
                forward_var = log_observation[i, :].clone()
                forward_var_all = log_observation[i, :]
            forward_var[~token_mask[gt_seq[i]]] = -np.inf
        return safe_logsumexp(forward_var_all, dim=0) \
            - safe_logsumexp(forward_var, dim=0)
