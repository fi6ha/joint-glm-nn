"""
main.py
-------
Full training pipeline for all five frequency-severity model variants.

Paper: "A Joint GLM-Neural Network for Frequency-Severity Modeling:
        Empirical Evaluation on French Motor Insurance Data"

Run order:
    1. python main.py          (trains all models, saves model_4b.pth)
    2. python shap_analysis.py (loads model_4b.pth, generates SHAP figures)

Dataset:
    freMTPL2freq.csv and freMTPL2sev.csv
    Download instructions in data/README.md
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models import SharedTrunkModel, ResidualNNModel, BayesianDropoutModel, SeverityOnlyNN
from utils import (
    to_log_sev, from_log_sev,
    poisson_loss, masked_mse, masked_gamma,
    make_dataloader, recalibrate
)

# ─────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────

ALPHA    = 1.0   # frequency loss weight
BETA     = 2.0   # severity loss weight
N_EPOCHS = 100
LR       = 0.0003
BATCH    = 512


# ─────────────────────────────────────────────────────────
# STEP 1: LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading and preparing data")
print("=" * 60)

df_freq = pd.read_csv("data/freMTPL2freq.csv")
df_sev  = pd.read_csv("data/freMTPL2sev.csv")

sev_agg = df_sev.groupby("IDpol", as_index=False).agg(
    TotalClaimAmount=("ClaimAmount", "sum"),
    NumClaims_sev=("ClaimAmount", "count")
)

df = pd.merge(df_freq, sev_agg, on="IDpol", how="left")
df["TotalClaimAmount"] = df["TotalClaimAmount"].fillna(0)
df["ClaimNb"]          = df["ClaimNb"].fillna(0).astype(int)

mask_claims = df["ClaimNb"] > 0
df["Severity_per_claim"] = 0.0
df.loc[mask_claims, "Severity_per_claim"] = (
    df.loc[mask_claims, "TotalClaimAmount"] / df.loc[mask_claims, "ClaimNb"]
)
df["ActualTotalClaim"] = df["TotalClaimAmount"]

numeric_features     = ["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
categorical_features = ["VehBrand", "VehGas", "Area", "Region"]

df_dummies = pd.get_dummies(df[categorical_features], drop_first=True)
df_dummies.columns = (
    df_dummies.columns
    .str.replace("'", "", regex=False)
    .str.replace("-", "_", regex=False)
    .str.replace(" ", "_", regex=False)
)

df_model = pd.concat(
    [df[numeric_features + ["IDpol", "ClaimNb", "Severity_per_claim", "ActualTotalClaim"]],
     df_dummies],
    axis=1
)
df_model = df_model.loc[:, ~df_model.columns.duplicated()]

features = [c for c in df_model.columns
            if c not in ["IDpol", "ClaimNb", "Severity_per_claim",
                         "ActualTotalClaim", "Exposure_safe"]]

train_pol, test_pol = train_test_split(
    df_model["IDpol"].unique(), test_size=0.2, random_state=SEED
)
train_full = df_model[df_model["IDpol"].isin(train_pol)].reset_index(drop=True)
test       = df_model[df_model["IDpol"].isin(test_pol)].reset_index(drop=True)

train_idx, val_idx = train_test_split(range(len(train_full)), test_size=0.1, random_state=SEED)
train = train_full.iloc[train_idx].reset_index(drop=True)
val   = train_full.iloc[val_idx].reset_index(drop=True)

for df_ in [train, val, test]:
    df_["Exposure_safe"] = df_["Exposure"].replace(0, 1e-8)

test_c = test[test["ClaimNb"] > 0].copy()

print(f"Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")
print(f"Claim rate: {(train['ClaimNb'] > 0).mean():.3%}")


# ─────────────────────────────────────────────────────────
# STEP 2: GLM POISSON + GLM GAMMA BASELINES
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: GLM baselines")
print("=" * 60)

glm_features = [f for f in features if f != "Exposure"]
formula      = "ClaimNb ~ " + " + ".join(glm_features)

glm = smf.glm(
    formula=formula,
    data=train,
    family=sm.families.Poisson(),
    offset=np.log(train["Exposure_safe"])
).fit()

train["glm_freq"] = glm.predict(train, offset=np.log(train["Exposure_safe"]))
val["glm_freq"]   = glm.predict(val,   offset=np.log(val["Exposure_safe"]))
test["glm_freq"]  = glm.predict(test,  offset=np.log(test["Exposure_safe"]))

glm_mse = mean_squared_error(test["ClaimNb"], test["glm_freq"])
glm_mae = mean_absolute_error(test["ClaimNb"], test["glm_freq"])
print(f"GLM Poisson  —  Freq MSE: {glm_mse:.6f} | MAE: {glm_mae:.6f}")

train_c_glm   = train[train["ClaimNb"] > 0].copy()
glm_gamma     = smf.glm(
    "Severity_per_claim ~ " + " + ".join(glm_features),
    data=train_c_glm,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()
test_c["glm_gamma_sev"] = glm_gamma.predict(test_c)
glm_gamma_mae = mean_absolute_error(test_c["Severity_per_claim"], test_c["glm_gamma_sev"])
print(f"GLM Gamma    —  Sev MAE:  {glm_gamma_mae:.2f}")


# ─────────────────────────────────────────────────────────
# STEP 3: FEATURE SETUP AND TENSORS
# ─────────────────────────────────────────────────────────

# Base features — Options 1, 2, 3, 4a
scaler_base  = StandardScaler()
X_train_base = scaler_base.fit_transform(train[features].values)
X_val_base   = scaler_base.transform(val[features].values)
X_test_base  = scaler_base.transform(test[features].values)

# Augmented features — Option 4b (GLM prediction concatenated)
for df_ in [train, val, test]:
    df_["log_glm_freq"] = np.log(df_["glm_freq"] + 1e-8)
augmented_features = features + ["glm_freq", "log_glm_freq"]

scaler_aug  = StandardScaler()
X_train_aug = scaler_aug.fit_transform(train[augmented_features].values)
X_val_aug   = scaler_aug.transform(val[augmented_features].values)
X_test_aug  = scaler_aug.transform(test[augmented_features].values)

# Severity targets
y_sev_train_log = to_log_sev(train["Severity_per_claim"].values)
y_sev_val_log   = to_log_sev(val["Severity_per_claim"].values)
y_sev_train_raw = train["Severity_per_claim"].values
y_sev_val_raw   = val["Severity_per_claim"].values

# Tensors
X_train_base_t = torch.tensor(X_train_base, dtype=torch.float32)
X_val_base_t   = torch.tensor(X_val_base,   dtype=torch.float32)
X_test_base_t  = torch.tensor(X_test_base,  dtype=torch.float32)
X_train_aug_t  = torch.tensor(X_train_aug,  dtype=torch.float32)
X_val_aug_t    = torch.tensor(X_val_aug,    dtype=torch.float32)
X_test_aug_t   = torch.tensor(X_test_aug,   dtype=torch.float32)

y_freq_tr    = torch.tensor(train["ClaimNb"].values, dtype=torch.float32).view(-1, 1)
y_freq_va    = torch.tensor(val["ClaimNb"].values,   dtype=torch.float32).view(-1, 1)
y_sev_log_tr = torch.tensor(y_sev_train_log,         dtype=torch.float32).view(-1, 1)
y_sev_log_va = torch.tensor(y_sev_val_log,           dtype=torch.float32).view(-1, 1)
y_sev_raw_tr = torch.tensor(y_sev_train_raw,         dtype=torch.float32).view(-1, 1)
y_sev_raw_va = torch.tensor(y_sev_val_raw,           dtype=torch.float32).view(-1, 1)

mask_tr = torch.tensor(train["ClaimNb"].values > 0, dtype=torch.bool)
mask_va = torch.tensor(val["ClaimNb"].values > 0,   dtype=torch.bool)

# GLM residuals for Option 2
glm_resid_train = train["ClaimNb"].values - train["glm_freq"].values
glm_resid_val   = val["ClaimNb"].values   - val["glm_freq"].values
y_resid_tr = torch.tensor(glm_resid_train, dtype=torch.float32).view(-1, 1)
y_resid_va = torch.tensor(glm_resid_val,   dtype=torch.float32).view(-1, 1)

train_c_nn = train[train["ClaimNb"] > 0].reset_index(drop=True)
val_c_nn   = val[val["ClaimNb"] > 0].reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# OPTION 1: GLM + SEPARATE NN
# Frequency: GLM Poisson (standalone)
# Severity:  separate NN trained only on claim policies
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OPTION 1: GLM + Separate NN")
print("=" * 60)

sc1      = StandardScaler()
Xtr_sev1 = sc1.fit_transform(train_c_nn[features].values)
Xva_sev1 = sc1.transform(val_c_nn[features].values)
Xte_sev1 = sc1.transform(test_c[features].values)

ytr_sev1 = torch.tensor(to_log_sev(train_c_nn["Severity_per_claim"].values),
                         dtype=torch.float32).view(-1, 1)
yva_sev1 = torch.tensor(to_log_sev(val_c_nn["Severity_per_claim"].values),
                         dtype=torch.float32).view(-1, 1)
Xtr1_t   = torch.tensor(Xtr_sev1, dtype=torch.float32)
Xva1_t   = torch.tensor(Xva_sev1, dtype=torch.float32)
Xte1_t   = torch.tensor(Xte_sev1, dtype=torch.float32)

from torch.utils.data import DataLoader, TensorDataset

sev_nn1 = SeverityOnlyNN(Xtr1_t.shape[1])
opt_s1  = torch.optim.Adam(sev_nn1.parameters(), lr=LR, weight_decay=1e-5)
sch_s1  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_s1, patience=5, factor=0.5)
dl_sev1 = DataLoader(TensorDataset(Xtr1_t, ytr_sev1), batch_size=256, shuffle=True)

for epoch in range(N_EPOCHS):
    sev_nn1.train()
    for Xb, ys in dl_sev1:
        opt_s1.zero_grad()
        F.mse_loss(sev_nn1(Xb), ys).backward()
        opt_s1.step()
    sev_nn1.eval()
    with torch.no_grad():
        sch_s1.step(F.mse_loss(sev_nn1(Xva1_t), yva_sev1))
    if (epoch + 1) % 20 == 0:
        print(f"  [Opt1] Epoch {epoch+1}/{N_EPOCHS}")

sev_nn1.eval()
with torch.no_grad():
    pred_logsev1 = sev_nn1(Xte1_t).numpy().flatten()

pred_sev1_raw = from_log_sev(pred_logsev1)
pred_sev1     = recalibrate(pred_sev1_raw,
                             train_c_nn["Severity_per_claim"].values,
                             np.ones(len(pred_sev1_raw), dtype=bool))

opt1_freq_mse = glm_mse
opt1_freq_mae = glm_mae
opt1_sev_mae  = mean_absolute_error(test_c["Severity_per_claim"], pred_sev1)
opt1_mean_el  = (test["glm_freq"] * pred_sev1.mean()).mean()
print(f"Option 1  —  Freq MSE: {opt1_freq_mse:.6f} | Sev MAE: {opt1_sev_mae:.2f} | Mean EL: {opt1_mean_el:.2f}")


# ─────────────────────────────────────────────────────────
# OPTION 2: GLM + RESIDUAL NN
# Frequency: GLM + NN residual correction
# Severity:  NN head from shared trunk
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OPTION 2: GLM + Residual NN")
print("=" * 60)

resid_nn = ResidualNNModel(X_train_base_t.shape[1])
opt_r2   = torch.optim.Adam(resid_nn.parameters(), lr=LR, weight_decay=1e-5)
sch_r2   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_r2, patience=5, factor=0.5)
dl_r2    = make_dataloader(X_train_base_t, y_resid_tr, y_sev_log_tr, mask_tr)

for epoch in range(N_EPOCHS):
    resid_nn.train()
    for Xb, yr, ys, cm in dl_r2:
        opt_r2.zero_grad()
        pr, ps = resid_nn(Xb)
        loss   = ALPHA * F.mse_loss(pr, yr) + BETA * masked_mse(ps, ys, cm.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(resid_nn.parameters(), 1.0)
        opt_r2.step()
    resid_nn.eval()
    with torch.no_grad():
        vr, vs = resid_nn(X_val_base_t)
        v_loss = ALPHA * F.mse_loss(vr, y_resid_va) + BETA * masked_mse(vs, y_sev_log_va, mask_va)
    sch_r2.step(v_loss)
    if (epoch + 1) % 20 == 0:
        print(f"  [Opt2] Epoch {epoch+1}/{N_EPOCHS} — Val: {v_loss.item():.4f}")

resid_nn.eval()
with torch.no_grad():
    pred_resid2, pred_ls2 = resid_nn(X_test_base_t)

pred_freq2    = np.clip(test["glm_freq"].values + pred_resid2.numpy().flatten(), 0, None)
pred_sev2_raw = from_log_sev(pred_ls2.numpy().flatten())
pred_sev2     = recalibrate(pred_sev2_raw,
                             train_c_nn["Severity_per_claim"].values,
                             test["ClaimNb"].values > 0)

opt2_freq_mse = mean_squared_error(test["ClaimNb"], pred_freq2)
opt2_freq_mae = mean_absolute_error(test["ClaimNb"], pred_freq2)
opt2_sev_mae  = mean_absolute_error(test_c["Severity_per_claim"],
                                    pred_sev2[test["ClaimNb"].values > 0])
opt2_mean_el  = (pred_freq2 * pred_sev2).mean()
print(f"Option 2  —  Freq MSE: {opt2_freq_mse:.6f} | Sev MAE: {opt2_sev_mae:.2f} | Mean EL: {opt2_mean_el:.2f}")


# ─────────────────────────────────────────────────────────
# OPTION 3: GLM + BAYESIAN NN (MC Dropout)
# Frequency: GLM Poisson (standalone)
# Severity:  Bayesian NN with dropout active at inference
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OPTION 3: GLM + Bayesian NN (MC Dropout)")
print("=" * 60)

bayes_nn = BayesianDropoutModel(X_train_base_t.shape[1])
opt_b3   = torch.optim.Adam(bayes_nn.parameters(), lr=LR, weight_decay=1e-5)
sch_b3   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_b3, patience=5, factor=0.5)
dl_b3    = make_dataloader(X_train_base_t, y_freq_tr, y_sev_log_tr, mask_tr)

for epoch in range(N_EPOCHS):
    bayes_nn.train()
    for Xb, yf, ys, cm in dl_b3:
        opt_b3.zero_grad()
        pf, ps = bayes_nn(Xb)
        loss   = ALPHA * poisson_loss(pf, yf) + BETA * masked_mse(ps, ys, cm.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bayes_nn.parameters(), 1.0)
        opt_b3.step()
    bayes_nn.eval()
    with torch.no_grad():
        vf, vs = bayes_nn(X_val_base_t)
        v_loss = ALPHA * poisson_loss(vf, y_freq_va) + BETA * masked_mse(vs, y_sev_log_va, mask_va)
    sch_b3.step(v_loss)
    if (epoch + 1) % 20 == 0:
        print(f"  [Opt3] Epoch {epoch+1}/{N_EPOCHS} — Val: {v_loss.item():.4f}")

freq_mean3, sev_mean3, freq_std3 = bayes_nn.predict_with_uncertainty(X_test_base_t, n_samples=30)

pred_sev3_raw = from_log_sev(sev_mean3)
pred_sev3     = recalibrate(pred_sev3_raw,
                             train_c_nn["Severity_per_claim"].values,
                             test["ClaimNb"].values > 0)

opt3_freq_mse = glm_mse
opt3_freq_mae = glm_mae
opt3_sev_mae  = mean_absolute_error(test_c["Severity_per_claim"],
                                    pred_sev3[test["ClaimNb"].values > 0])
opt3_mean_el  = (test["glm_freq"].values * pred_sev3).mean()
print(f"Option 3  —  Freq MSE: {opt3_freq_mse:.6f} | Sev MAE: {opt3_sev_mae:.2f} | Mean EL: {opt3_mean_el:.2f}")
print(f"             Mean freq uncertainty (std): {freq_std3.mean():.5f}")


# ─────────────────────────────────────────────────────────
# OPTION 4a: JOINT NN — NO GLM INPUT
# Shared trunk, MSE on log-severity
# Included to demonstrate calibration collapse without Gamma deviance
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OPTION 4a: Joint NN (no GLM input, MSE on log-severity)")
print("=" * 60)

model_4a = SharedTrunkModel(X_train_base_t.shape[1])
opt_4a   = torch.optim.Adam(model_4a.parameters(), lr=LR, weight_decay=1e-5)
sch_4a   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_4a, patience=5, factor=0.5)
dl_4a    = make_dataloader(X_train_base_t, y_freq_tr, y_sev_log_tr, mask_tr)
hist_4a  = {"train": [], "val": []}

for epoch in range(N_EPOCHS):
    model_4a.train()
    ep_loss, ep_n = 0, 0
    for Xb, yf, ys, cm in dl_4a:
        opt_4a.zero_grad()
        pf, ps = model_4a(Xb)
        loss   = ALPHA * poisson_loss(pf, yf) + BETA * masked_mse(ps, ys, cm.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_4a.parameters(), 1.0)
        opt_4a.step()
        ep_loss += loss.item(); ep_n += 1
    model_4a.eval()
    with torch.no_grad():
        vf, vs = model_4a(X_val_base_t)
        v_loss = ALPHA * poisson_loss(vf, y_freq_va) + BETA * masked_mse(vs, y_sev_log_va, mask_va)
    hist_4a["train"].append(ep_loss / ep_n)
    hist_4a["val"].append(v_loss.item())
    sch_4a.step(v_loss)
    if (epoch + 1) % 20 == 0:
        print(f"  [Opt4a] Epoch {epoch+1}/{N_EPOCHS} — Val: {v_loss.item():.4f}")

model_4a.eval()
with torch.no_grad():
    pf_4a, ps_4a = model_4a(X_test_base_t)

pred_freq_4a    = pf_4a.numpy().flatten()
pred_sev_4a_raw = from_log_sev(ps_4a.numpy().flatten())

test["pred_4a_freq"] = pred_freq_4a
test["pred_4a_sev"]  = pred_sev_4a_raw
test["pred_4a_loss"] = pred_freq_4a * pred_sev_4a_raw

opt4a_freq_mse  = mean_squared_error(test["ClaimNb"], pred_freq_4a)
opt4a_freq_mae  = mean_absolute_error(test["ClaimNb"], pred_freq_4a)
opt4a_sev_mae   = mean_absolute_error(test_c["Severity_per_claim"],
                                      pred_sev_4a_raw[test["ClaimNb"].values > 0])
opt4a_mean_el   = test["pred_4a_loss"].mean()
opt4a_cal_ratio = test["pred_4a_loss"].sum() / test["ActualTotalClaim"].sum()
print(f"Option 4a —  Freq MSE: {opt4a_freq_mse:.6f} | Sev MAE: {opt4a_sev_mae:.2f} | "
      f"Mean EL: {opt4a_mean_el:.2f} | Cal Ratio: {opt4a_cal_ratio:.3f}")


# ─────────────────────────────────────────────────────────
# OPTION 4b: GLM-INFORMED JOINT NN — FINAL MODEL
# Shared trunk + GLM prediction concatenated to input
# Gamma deviance on raw severity (fixes 4a calibration collapse)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OPTION 4b: GLM-Informed Joint NN [FINAL]")
print("=" * 60)

model_4b = SharedTrunkModel(X_train_aug_t.shape[1])
opt_4b   = torch.optim.Adam(model_4b.parameters(), lr=LR, weight_decay=1e-5)
sch_4b   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_4b, patience=5, factor=0.5)
dl_4b    = make_dataloader(X_train_aug_t, y_freq_tr, y_sev_raw_tr, mask_tr)
hist_4b  = {"train": [], "val": []}

for epoch in range(N_EPOCHS):
    model_4b.train()
    ep_loss, ep_n = 0, 0
    for Xb, yf, ys, cm in dl_4b:
        opt_4b.zero_grad()
        pf, ps = model_4b(Xb)
        loss   = ALPHA * poisson_loss(pf, yf) + BETA * masked_gamma(ps, ys, cm.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_4b.parameters(), 1.0)
        opt_4b.step()
        ep_loss += loss.item(); ep_n += 1
    model_4b.eval()
    with torch.no_grad():
        vf, vs = model_4b(X_val_aug_t)
        v_loss = ALPHA * poisson_loss(vf, y_freq_va) + BETA * masked_gamma(vs, y_sev_raw_va, mask_va)
    hist_4b["train"].append(ep_loss / ep_n)
    hist_4b["val"].append(v_loss.item())
    sch_4b.step(v_loss)
    if (epoch + 1) % 20 == 0:
        print(f"  [Opt4b] Epoch {epoch+1}/{N_EPOCHS} — "
              f"Train: {hist_4b['train'][-1]:.4f} | Val: {hist_4b['val'][-1]:.4f}")

model_4b.eval()
with torch.no_grad():
    pf_4b, ps_4b = model_4b(X_test_aug_t)

pred_freq_4b    = pf_4b.numpy().flatten()
pred_sev_4b_raw = (F.softplus(ps_4b) + 1.0).numpy().flatten()
pred_sev_4b     = recalibrate(pred_sev_4b_raw,
                               train[train["ClaimNb"] > 0]["Severity_per_claim"].values,
                               test["ClaimNb"].values > 0)

test["pred_4b_freq"] = pred_freq_4b
test["pred_4b_sev"]  = pred_sev_4b
test["pred_4b_loss"] = pred_freq_4b * pred_sev_4b

opt4b_freq_mse  = mean_squared_error(test["ClaimNb"], pred_freq_4b)
opt4b_freq_mae  = mean_absolute_error(test["ClaimNb"], pred_freq_4b)
opt4b_sev_mae   = mean_absolute_error(test_c["Severity_per_claim"],
                                      pred_sev_4b[test["ClaimNb"].values > 0])
opt4b_mean_el   = test["pred_4b_loss"].mean()
opt4b_cal_ratio = test["pred_4b_loss"].sum() / test["ActualTotalClaim"].sum()
print(f"Option 4b —  Freq MSE: {opt4b_freq_mse:.6f} | Sev MAE: {opt4b_sev_mae:.2f} | "
      f"Mean EL: {opt4b_mean_el:.2f} | Cal Ratio: {opt4b_cal_ratio:.3f}")

torch.save(model_4b.state_dict(), "model_4b.pth")
print("Model saved: model_4b.pth")


# ─────────────────────────────────────────────────────────
# FULL MODEL COMPARISON TABLE
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FULL COMPARISON — ALL MODEL OPTIONS")
print("=" * 70)

comparison = pd.DataFrame({
    "Model": [
        "Option 1: GLM + Separate NN",
        "Option 2: GLM + Residual NN",
        "Option 3: GLM + Bayesian NN",
        "Option 4a: Joint NN (no GLM input)",
        "Option 4b: GLM-Informed Joint NN  [FINAL]"
    ],
    "Freq_MSE":       [opt1_freq_mse,  opt2_freq_mse,  opt3_freq_mse,  opt4a_freq_mse,  opt4b_freq_mse],
    "Sev_MAE":        [opt1_sev_mae,   opt2_sev_mae,   opt3_sev_mae,   opt4a_sev_mae,   opt4b_sev_mae],
    "Mean_EL":        [opt1_mean_el,   opt2_mean_el,   opt3_mean_el,   opt4a_mean_el,   opt4b_mean_el],
    "Cal_Ratio":      ["n/a", "n/a", "n/a", f"{opt4a_cal_ratio:.3f}", f"{opt4b_cal_ratio:.3f}"],
    "Joint_Learning": ["No", "Partial", "No", "Yes", "Yes"],
    "GLM_Input":      ["No", "Yes", "No", "No", "Yes"]
})

pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 150)
print(comparison.to_string(index=False))
comparison.to_csv("final_model_comparison.csv", index=False)
print("\nComparison saved: final_model_comparison.csv")


# ─────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────
print("\nGenerating plots")

epochs_x = range(1, N_EPOCHS + 1)
fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs_x, hist_4b["train"], label="Train", color="navy",  lw=2)
ax1.plot(epochs_x, hist_4b["val"],   label="Val",   color="coral", lw=2, ls="--")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Total Loss")
ax1.set_title("Option 4b: Train vs Val Loss", fontsize=10)
ax1.legend(fontsize=8); ax1.grid(True, ls="--", lw=0.4)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs_x, hist_4a["val"], label="4a (no GLM input)", color="tomato",    lw=2)
ax2.plot(epochs_x, hist_4b["val"], label="4b (GLM input)",    color="steelblue", lw=2)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation Loss")
ax2.set_title("4a vs 4b: Validation Loss", fontsize=10)
ax2.legend(fontsize=8); ax2.grid(True, ls="--", lw=0.4)

ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(test["glm_freq"], test["pred_4b_freq"], alpha=0.2, s=8, color="steelblue")
lim = [0, test["glm_freq"].quantile(0.99)]
ax3.plot(lim, lim, "r--", lw=1.5, label="y = x")
ax3.set_xlabel("GLM Frequency"); ax3.set_ylabel("Option 4b Frequency")
ax3.set_title("GLM vs Hybrid Frequency\n(deviation = NN refinement)", fontsize=10)
ax3.legend(fontsize=8); ax3.grid(True, ls="--", lw=0.4)

ax4 = fig.add_subplot(gs[1, :2])
sc = ax4.scatter(
    test["pred_4b_freq"] + 1e-6,
    test["pred_4b_sev"]  + 1e-6,
    c=test["pred_4b_loss"] + 1e-6,
    cmap="viridis", alpha=0.5, s=12
)
ax4.set_xscale("log"); ax4.set_yscale("log")
plt.colorbar(sc, ax=ax4, label="Expected Loss (Freq x Sev)")
ax4.set_xlabel("Predicted Frequency (Option 4b)")
ax4.set_ylabel("Predicted Severity (Option 4b)")
ax4.set_title("Frequency vs Severity — Option 4b\nInverse relationship from shared trunk", fontsize=10)
ax4.grid(True, which="both", ls="--", lw=0.4)

test["freq_decile"] = pd.qcut(test["pred_4b_freq"], q=10, labels=False, duplicates="drop")
decile_cal = test.groupby("freq_decile").agg(
    actual_mean=("ActualTotalClaim", "mean"),
    predicted_mean=("pred_4b_loss", "mean"),
    n=("IDpol", "count")
).reset_index()

ax5 = fig.add_subplot(gs[1, 2])
x = decile_cal["freq_decile"]
ax5.bar(x - 0.2, decile_cal["actual_mean"],    width=0.4, label="Actual",    color="#FFD580", edgecolor="grey")
ax5.bar(x + 0.2, decile_cal["predicted_mean"], width=0.4, label="Predicted", color="#C7CEEA", edgecolor="grey")
ax5.set_xlabel("Risk Decile"); ax5.set_ylabel("Mean Claim Amount (£)")
ax5.set_title(f"Calibration by Risk Decile\n(ratio = {opt4b_cal_ratio:.3f})", fontsize=10)
ax5.legend(fontsize=8); ax5.grid(True, axis="y", ls="--", lw=0.4)

ax6 = fig.add_subplot(gs[2, 0])
cap = 30000
ax6.scatter(
    test_c["Severity_per_claim"].clip(upper=cap),
    pred_sev_4b[test["ClaimNb"].values > 0].clip(max=cap),
    alpha=0.3, s=8, color="mediumpurple"
)
ax6.plot([0, cap], [0, cap], "r--", lw=1.5, label="Perfect")
ax6.set_xlabel("Actual Severity (£)"); ax6.set_ylabel("Predicted Severity (£)")
ax6.set_title("Option 4b: Severity Actual vs Predicted\n(claim policies, capped £30k)", fontsize=10)
ax6.legend(fontsize=8); ax6.grid(True, ls="--", lw=0.4)

ax7 = fig.add_subplot(gs[2, 1])
model_names = ["Opt1\nSep.", "Opt2\nResid.", "Opt3\nBayes", "Opt4a\nJoint", "Opt4b\nFINAL"]
sev_maes    = [opt1_sev_mae, opt2_sev_mae, opt3_sev_mae, opt4a_sev_mae, opt4b_sev_mae]
bar_colors  = ["#FFD580", "#B5EAD7", "#C7CEEA", "#FFAAAA", "#90EE90"]
bars7 = ax7.bar(model_names, sev_maes, color=bar_colors, edgecolor="grey", lw=0.5)
for bar, v in zip(bars7, sev_maes):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax7.set_ylabel("Severity MAE (£)")
ax7.set_title("Severity MAE Across All Models", fontsize=10)
ax7.grid(True, axis="y", ls="--", lw=0.4)

ax8 = fig.add_subplot(gs[2, 2])
freq_mses = [opt1_freq_mse, opt2_freq_mse, opt3_freq_mse, opt4a_freq_mse, opt4b_freq_mse]
bars8 = ax8.bar(model_names, freq_mses, color=bar_colors, edgecolor="grey", lw=0.5)
for bar, v in zip(bars8, freq_mses):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
             f"{v:.5f}", ha="center", va="bottom", fontsize=7)
ax8.set_ylabel("Frequency MSE")
ax8.set_title("Frequency MSE Across All Models", fontsize=10)
ax8.grid(True, axis="y", ls="--", lw=0.4)

fig.suptitle(
    "Hybrid GLM-Informed Joint Frequency-Severity Model: All Options\n",
    fontsize=13, fontweight="bold"
)

plt.savefig("hybrid_model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: hybrid_model_results.png")
print("\nDone. Run shap_analysis.py next to generate SHAP figures.")
