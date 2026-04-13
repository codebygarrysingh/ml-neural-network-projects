# Neural Network Engineering — Applied Deep Learning Projects

> End-to-end deep learning implementations using PyTorch and TensorFlow: LSTM time-series forecasting, multi-class classification, and sensor data modelling. Focus on production-quality code, rigorous evaluation, and sound ML engineering practices.

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch_2.3-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_2.16-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

---

## Projects

### 1. LSTM Room Occupancy Estimator

**Problem:** Predict occupancy (0–3 people) from multivariate IoT sensor time series — CO₂, temperature, humidity, light, and sound — with irregular temporal patterns and missing data.

**Why LSTM?** Occupancy is a function of recent history, not just the current reading. A door opening 30 seconds ago is more predictive than current CO₂. LSTM's gating mechanism (forget/input/output gates) captures this temporal dependency; feedforward networks cannot.

**Architecture:**

```
Input: [batch, seq_len=60, features=5]  ← 60-step lookback window
         │
┌────────▼────────────────────────────┐
│  LSTM Layer                         │
│  hidden_size=128, num_layers=2      │
│  dropout=0.2 (between layers)       │
│  bidirectional=False                │
└────────┬────────────────────────────┘
         │  → hidden state h_t (last step only)
┌────────▼────────────────────────────┐
│  Dropout(0.3)                       │
└────────┬────────────────────────────┘
         │
┌────────▼────────────────────────────┐
│  Linear(128 → 64) + ReLU            │
└────────┬────────────────────────────┘
         │
┌────────▼────────────────────────────┐
│  Linear(64 → 4)                     │
│  (4 occupancy classes: 0,1,2,3)     │
└─────────────────────────────────────┘

Loss: CrossEntropyLoss
Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
Scheduler: ReduceLROnPlateau (patience=5)
```

**Engineering Highlights:**
- Sliding window dataset construction with configurable `seq_len` and `stride`
- StandardScaler fit on training split only — zero data leakage into validation
- Class-weighted loss for imbalanced occupancy distribution
- Gradient clipping (`max_norm=1.0`) for stable LSTM training

**Results:**

| Metric | Value |
|---|---|
| Test Accuracy | 94.2% |
| Macro F1 | 0.91 |
| MSE (raw regression formulation) | 0.043 |

---

### 2. Multi-Class Classification — Sensor Anomaly Detection

**Problem:** Classify sensor states across multiple environmental conditions — applicable to industrial IoT, building management, and financial market microstructure anomaly detection.

**Architecture:** Fully connected feed-forward network with batch normalisation:

```
Input Features → Linear(n→256) → BN → ReLU → Dropout(0.3)
              → Linear(256→128) → BN → ReLU → Dropout(0.3)
              → Linear(128→64)  → BN → ReLU
              → Linear(64→num_classes)
              → Softmax
```

**Key Decisions:**
- Batch normalisation between each hidden layer — stabilises training on heterogeneous sensor scales
- Dropout before final layer only — preserves feature learning capacity in early layers
- Optuna hyperparameter search over learning rate, hidden dims, and dropout rates

---

### 3. Regression Baseline Comparison

Systematic comparison of neural regression vs. gradient boosting on structured tabular data — establishing when deep learning adds value over classical ML:

| Model | RMSE | R² | Training Time |
|---|---|---|---|
| Linear Regression | 0.312 | 0.74 | <1s |
| Gradient Boosting (XGBoost) | 0.198 | 0.87 | 8s |
| Random Forest | 0.211 | 0.85 | 12s |
| Neural Network (3-layer MLP) | 0.183 | 0.89 | 45s |
| Neural Network + Feature Eng. | **0.161** | **0.92** | 52s |

**Takeaway:** Neural networks win on this dataset only with domain-specific feature engineering. Raw tabular data generally favours gradient boosting.

---

## ML Engineering Practices

All notebooks follow production-quality engineering standards:

```python
# Reproducibility — always set seeds before any random operations
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Evaluation — always report multiple metrics, never just accuracy
def evaluate(model, loader, device):
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            preds.extend(model(X.to(device)).argmax(1).cpu().numpy())
            targets.extend(y.numpy())
    return {
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro"),
        "confusion_matrix": confusion_matrix(targets, preds),
    }
```

---

## Repo Structure

```
ml-neural-network-projects/
├── lstm_occupancy_estimator/
│   ├── dataset.py          # Sliding window dataset, train/val/test split
│   ├── model.py            # LSTM architecture + weight initialisation
│   ├── train.py            # Training loop with early stopping
│   ├── evaluate.py         # Metrics, confusion matrix, loss curves
│   └── notebook.ipynb      # Full walkthrough with visualisations
├── classification_anomaly/
│   ├── model.py
│   ├── hyperparam_search.py  # Optuna integration
│   └── notebook.ipynb
└── regression_comparison/
    └── notebook.ipynb
```

---

## Related Work

- [ml-regression-python-projects](https://github.com/codebygarrysingh/ml-regression-python-projects) — classical regression methods
- [ml-classification-projects](https://github.com/codebygarrysingh/ml-classification-projects) — traditional ML classifiers
- [production-rag-pipeline](https://github.com/codebygarrysingh/production-rag-pipeline) — LLM engineering built on similar ML foundations

---

## Author

**Garry Singh** — Principal AI & Data Engineer · MSc Oxford

[Portfolio](https://garrysingh.dev) · [LinkedIn](https://linkedin.com/in/singhgarry) · [Book a Consultation](https://calendly.com/garry-singh2902)
