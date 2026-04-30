"""
models.py
---------
Neural network architectures for all five model variants.

Options:
  1  - GLM + Separate NN       (SeverityOnlyNN)
  2  - GLM + Residual NN       (ResidualNNModel)
  3  - GLM + Bayesian NN       (BayesianDropoutModel)
  4a - Joint NN                (SharedTrunkModel, no GLM input)
  4b - GLM-Informed Joint NN   (SharedTrunkModel, with GLM input) [FINAL]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedTrunkModel(nn.Module):
    """
    Shared trunk with two output heads: frequency and severity.
    Used for Option 4a (no GLM input) and Option 4b (GLM input concatenated).

    Architecture:
        Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x 3 -> shared trunk
        Trunk -> freq_head -> softplus output (ensures non-negative frequency)
        Trunk -> sev_head  -> raw output (severity, transformed at prediction time)
    """
    def __init__(self, in_features, hidden=(128, 96, 64), dropout=0.2):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk     = nn.Sequential(*layers)
        self.freq_head = nn.Sequential(nn.Linear(prev, 64), nn.ReLU(), nn.Linear(64, 1))
        self.sev_head  = nn.Sequential(nn.Linear(prev, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        shared = self.trunk(x)
        return F.softplus(self.freq_head(shared)), self.sev_head(shared)


class ResidualNNModel(nn.Module):
    """
    Option 2: NN predicts GLM frequency residuals and severity jointly.
    The final frequency prediction is GLM output + NN residual correction.
    """
    def __init__(self, in_features, hidden=(128, 96, 64), dropout=0.2):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk      = nn.Sequential(*layers)
        self.resid_head = nn.Sequential(nn.Linear(prev, 64), nn.ReLU(), nn.Linear(64, 1))
        self.sev_head   = nn.Sequential(nn.Linear(prev, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        shared = self.trunk(x)
        return self.resid_head(shared), self.sev_head(shared)


class BayesianDropoutModel(nn.Module):
    """
    Option 3: MC Dropout model for uncertainty estimation.
    Dropout is kept active at inference time to produce stochastic predictions,
    which are averaged over multiple forward passes (Monte Carlo sampling).
    """
    def __init__(self, in_features, hidden=(128, 96, 64), dropout=0.2):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk     = nn.Sequential(*layers)
        self.freq_head = nn.Sequential(nn.Linear(prev, 64), nn.ReLU(), nn.Linear(64, 1))
        self.sev_head  = nn.Sequential(nn.Linear(prev, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        shared = self.trunk(x)
        return F.softplus(self.freq_head(shared)), self.sev_head(shared)

    def predict_with_uncertainty(self, x, n_samples=30):
        """Run n_samples stochastic forward passes and return mean and std."""
        self.train()  # keep dropout active
        freq_s, sev_s = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                f, s = self.forward(x)
                freq_s.append(f.numpy())
                sev_s.append(s.numpy())
        return (
            np.mean(freq_s, axis=0).flatten(),
            np.mean(sev_s,  axis=0).flatten(),
            np.std(freq_s,  axis=0).flatten()
        )


class SeverityOnlyNN(nn.Module):
    """
    Option 1: Standalone severity network trained only on claim policies.
    Frequency is handled separately by the GLM Poisson baseline.
    """
    def __init__(self, in_features, hidden=(128, 96, 64), dropout=0.2):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net  = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(prev, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.net(x))
