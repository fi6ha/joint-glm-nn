"""
shap_analysis.py
----------------
SHAP interpretability analysis for Option 4b (GLM-Informed Joint NN).

Run this after main.py has saved model_4b.pth.

Outputs:
    shap_frequency.png  — top 15 features for the frequency head
    shap_severity.png   — top 15 features for the severity head
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import statsmodels.api as sm

from models import SharedTrunkModel

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────
# WRAPPERS — expose a single output head for SHAP
# ─────────────────────────────────────────────────────────

class FreqWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        freq, _ = self.model(x)
        return freq

class SevWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        _, sev = self.model(x)
        return sev


# ─────────────────────────────────────────────────────────
# RELOAD DATA
# Must match main.py exactly to reproduce the same feature set
# ─────────────────────────────────────────────────────────

print("Loading data...")

df_freq = pd.read_csv("data/freMTPL2freq.csv")
df_sev  = pd.read_csv("data/freMTPL2sev.csv")

sev_agg = df_sev.groupby("IDpol", as_index=False).agg(
    TotalClaimAmount=("ClaimAmount", "sum"),
    NumClaims_sev=("ClaimAmount", "count")
)
df = pd.merge(df_freq, sev_agg, on="IDpol", how="left")
df["TotalClaimAmount"] = df["TotalClaimAmount"].fillna(0)
df["ClaimNb"]          = df["ClaimNb"].fillna(0).astype(int)

numeric_features     = ["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
categorical_features = ["VehBrand", "VehGas", "Area", "Region"]

df_dummies = pd.get_dummies(df[categorical_features], drop_first=True)
df_dummies.columns = (df_dummies.columns
    .str.replace("'", "", regex=False)
    .str.replace("-", "_", regex=False)
    .str.replace(" ", "_", regex=False))

df_model = pd.concat(
    [df[numeric_features + ["IDpol", "ClaimNb"]], df_dummies], axis=1)
df_model = df_model.loc[:, ~df_model.columns.duplicated()]

features = [c for c in df_model.columns
            if c not in ["IDpol", "ClaimNb", "Exposure_safe"]]

train_pol, test_pol = train_test_split(
    df_model["IDpol"].unique(), test_size=0.2, random_state=SEED)
train_full = df_model[df_model["IDpol"].isin(train_pol)].reset_index(drop=True)
test       = df_model[df_model["IDpol"].isin(test_pol)].reset_index(drop=True)

train_idx, _ = train_test_split(range(len(train_full)), test_size=0.1, random_state=SEED)
train = train_full.iloc[train_idx].reset_index(drop=True)

for df_ in [train, test]:
    df_["Exposure_safe"] = df_["Exposure"].replace(0, 1e-8)

# Refit GLM to get glm_freq (must match main.py)
glm_features = [f for f in features if f != "Exposure"]
formula      = "ClaimNb ~ " + " + ".join(glm_features)
glm = smf.glm(formula=formula, data=train,
              family=sm.families.Poisson(),
              offset=np.log(train["Exposure_safe"])).fit()

train["glm_freq"] = glm.predict(train, offset=np.log(train["Exposure_safe"]))
test["glm_freq"]  = glm.predict(test,  offset=np.log(test["Exposure_safe"]))

for df_ in [train, test]:
    df_["log_glm_freq"] = np.log(df_["glm_freq"] + 1e-8)

augmented_features = features + ["glm_freq", "log_glm_freq"]

scaler_aug  = StandardScaler()
X_train_aug = scaler_aug.fit_transform(train[augmented_features].values)
X_test_aug  = scaler_aug.transform(test[augmented_features].values)

X_train_aug_t = torch.tensor(X_train_aug, dtype=torch.float32)
X_test_aug_t  = torch.tensor(X_test_aug,  dtype=torch.float32)


# ─────────────────────────────────────────────────────────
# LOAD SAVED MODEL
# ─────────────────────────────────────────────────────────

print("Loading model_4b.pth...")
model_4b = SharedTrunkModel(X_train_aug_t.shape[1])
model_4b.load_state_dict(torch.load("model_4b.pth"))
model_4b.eval()

background  = X_train_aug_t[:100]
test_sample = X_test_aug_t[:1000]


# ─────────────────────────────────────────────────────────
# SHAP — FREQUENCY HEAD
# ─────────────────────────────────────────────────────────

print("Running SHAP for frequency head...")
explainer_freq = shap.DeepExplainer(FreqWrapper(model_4b), background)
shap_freq      = explainer_freq.shap_values(test_sample, check_additivity=False)
shap_freq_2d   = shap_freq.squeeze(-1)

plt.figure(figsize=(15, 10))
shap.summary_plot(shap_freq_2d, test_sample.numpy(),
                  feature_names=augmented_features,
                  max_display=15,
                  plot_type="bar",
                  show=False)
plt.title("SHAP Feature Importance: Frequency Head", pad=12)
plt.xlabel("Mean |SHAP Value|", fontsize=12)
ax = plt.gca()
ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
plt.tight_layout()
plt.savefig("shap_frequency.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_frequency.png")


# ─────────────────────────────────────────────────────────
# SHAP — SEVERITY HEAD
# ─────────────────────────────────────────────────────────

print("Running SHAP for severity head...")
explainer_sev = shap.DeepExplainer(SevWrapper(model_4b), background)
shap_sev      = explainer_sev.shap_values(test_sample, check_additivity=False)
shap_sev_2d   = shap_sev.squeeze(-1)

plt.figure(figsize=(15, 10))
shap.summary_plot(shap_sev_2d, test_sample.numpy(),
                  feature_names=augmented_features,
                  max_display=15,
                  plot_type="bar",
                  show=False)
plt.title("SHAP Feature Importance: Severity Head", pad=12)
plt.xlabel("Mean |SHAP Value|", fontsize=12)
ax = plt.gca()
ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
plt.tight_layout()
plt.savefig("shap_severity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_severity.png")

print("\nSHAP analysis complete.")
