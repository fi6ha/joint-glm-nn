# Data

This project uses the **freMTPL2** (French Motor Third-Party Liability) dataset, a standard benchmark for insurance pricing research.

## Download

The dataset is freely available and has no copyright restrictions for academic use.

**Option 1 — OpenML (recommended)**

```python
import openml
dataset = openml.datasets.get_dataset(41214)
```

Or download directly from: https://www.openml.org/d/41214

**Option 2: Kaggle**

Search for `freMTPL2` on Kaggle and download the CSV files.

## Setup

Place the two CSV files in this `data/` folder before running the code:

```
data/
├── freMTPL2freq.csv
├── freMTPL2sev.csv
└── README.md
```

## Dataset Description

| File | Rows | Description |
|------|------|-------------|
| freMTPL2freq.csv | 677,991 | Policy-level features and claim counts |
| freMTPL2sev.csv | ~26,000 | Individual claim amounts |

Key features: `Exposure`, `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `Density`, `VehBrand`, `VehGas`, `Area`, `Region`

Target variables: `ClaimNb` (frequency), `ClaimAmount` (severity)

## Reference

Dutang, C. (2024). CASdatasets: French Motor Third-Party Liability datasets. https://dutangc.github.io/CASdatasets/reference/freMTPL.html
