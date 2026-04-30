"""
utils.py
--------
Loss functions and helper utilities shared across all model variants.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────
# SEVERITY TRANSFORMS
# Log transform used for Options 1, 2, 3, 4a (MSE on log-severity)
# Option 4b uses raw severity with Gamma deviance instead
# ─────────────────────────────────────────────────────────

LOG_OFFSET = 1.0

def to_log_sev(x):
    """Log-transform severity targets (with offset to handle zeros)."""
    return np.log(x + LOG_OFFSET)

def from_log_sev(x):
    """Inverse log-transform: recover raw severity predictions."""
    return np.clip(np.exp(x) - LOG_OFFSET, 0, None)


# ─────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────

def poisson_loss(pred, true):
    """
    Poisson deviance loss for frequency modeling.
    More appropriate than MSE for count data.
    """
    return (pred - true * torch.log(pred + 1e-8)).mean()


def masked_mse(pred, true, mask):
    """
    MSE loss applied only to policies with at least one claim.
    Avoids penalising zero-severity predictions for non-claim policies.
    """
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return F.mse_loss(pred[mask], true[mask])


def masked_huber(pred, true, mask, delta=1.0):
    """
    Huber loss applied only to claim policies.
    More robust to large severity outliers than MSE.
    """
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return F.huber_loss(pred[mask], true[mask], delta=delta)


def masked_gamma(pred, true, mask):
    """
    Gamma deviance loss on raw severity — used in Option 4b.
    Penalises relative errors rather than absolute errors, which is more
    appropriate for right-skewed claim severity distributions.
    This loss fixes the calibration collapse observed in Option 4a (MSE + log).
    """
    if mask.sum() == 0:
        return torch.tensor(0.0)
    p = torch.clamp(F.softplus(pred[mask]) + 1.0, min=1.0)
    t = torch.clamp(true[mask], min=1.0)
    return (t / p - torch.log(t / p) - 1).mean()


# ─────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────

def make_dataloader(X_t, y_freq, y_sev, mask, batch_size=512):
    """
    Build a DataLoader from tensors for joint frequency-severity training.
    mask indicates which policies have at least one claim (used for severity loss).
    """
    return DataLoader(
        TensorDataset(X_t, y_freq, y_sev, mask.view(-1, 1)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )


def recalibrate(pred_raw, ref_values, mask_bool):
    """
    Scale severity predictions so the mean on claim policies matches
    the training set mean. Corrects systematic under or over prediction.
    """
    train_mean = ref_values.mean()
    pred_mean  = pred_raw[mask_bool].mean()
    if pred_mean == 0:
        return pred_raw
    return pred_raw * (train_mean / pred_mean)
