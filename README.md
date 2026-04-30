# Joint GLM-Neural Network for Frequency-Severity Insurance Modeling

Code repository for the paper:

> **"A Joint GLM-Neural Network for Frequency-Severity Modeling: Empirical Evaluation on French Motor Insurance Data"**
> Fidha Faisal, Dr. Andres Barajas Paz, Dr. Nasreddine Megrez - Heriot-Watt University (2026)

## Overview

This repository provides the full implementation of five frequency-severity model variants evaluated on the freMTPL2 benchmark dataset. The final model (Option 4b) is a GLM-informed joint neural network that concatenates a Poisson GLM frequency prediction to the input of a shared-trunk neural network, trained with Gamma deviance on raw severity.

**Key results on the freMTPL2 test set (n = 135,603 policies):**

| Model | Freq MSE | Sev MAE | Cal. Ratio |
|-------|----------|---------|------------|
| Option 1: GLM + Separate NN | 0.05633 | 1,880 | n/a |
| Option 2: GLM + Residual NN | 0.05572 | 1,843 | n/a |
| Option 3: GLM + Bayesian NN | 0.05633 | 1,867 | n/a |
| Option 4a: Joint NN | 0.05581 | 1,374 | 0.236 |
| **Option 4b: GLM-Informed Joint NN** | **0.05587** | **1,700** | **1.025** |

## Repository Structure

```
joint-glm-nn/
├── main.py               # Full training pipeline — all five model variants
├── models.py             # Neural network architectures
├── utils.py              # Loss functions and helper utilities
├── shap_analysis.py      # SHAP interpretability for Option 4b
├── results.ipynb         # Notebook with results and figures
├── requirements.txt      # Python dependencies
└── data/
    └── README.md         # Dataset download instructions
```

## Setup

```bash
git clone https://github.com/fi6ha/joint-glm-nn
cd joint-glm-nn
pip install -r requirements.txt
```

Download the freMTPL2 dataset and place it in the `data/` folder. See `data/README.md` for instructions.

## Running the Code

**Step 1: Train all models and generate figures:**
```bash
python main.py
```

This will print results for all five options, save `model_4b.pth`, `final_model_comparison.csv`, and `hybrid_model_results.png`.

**Step 2: Run SHAP interpretability analysis:**
```bash
python shap_analysis.py
```

This loads the saved model and generates `shap_frequency.png` and `shap_severity.png`.

**Alternatively: open the notebook:**
```bash
jupyter notebook results.ipynb
```

## Model Architecture (Option 4b)

```
Input features (base + glm_freq + log_glm_freq)
        |
[Linear(128) → BatchNorm → ReLU → Dropout(0.2)]
[Linear(96)  → BatchNorm → ReLU → Dropout(0.2)]
[Linear(64)  → BatchNorm → ReLU → Dropout(0.2)]
        |
   Shared Trunk
      /       \
Freq Head    Sev Head
(softplus)   (softplus + 1)
```

Training: Adam (lr=0.0003), 100 epochs, Poisson loss (frequency) + Gamma deviance (severity).

## Citation

If you use this code, please cite:

```
@article{faisal2026joint,
  title={A Joint GLM-Neural Network for Frequency-Severity Modeling: 
         Empirical Evaluation on French Motor Insurance Data},
  author={Faisal, Fidha and Barajas Paz, Andres and Megrez, Nasreddine},
  institution={Heriot-Watt University},
  year={2026}
}
```
