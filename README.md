# 🧩 PrivBench — Unified Benchmarking Framework for Differential Privacy in ML

**PrivBench** is an open-source benchmarking framework that makes the **privacy–utility trade-off in machine learning measurable, transparent, and reproducible**.  
It extends traditional Differential Privacy (DP) evaluation by introducing four standardized dimensions: **Privacy Strength**, **Model Utility**, **Fairness**, and **Efficiency**.

---

## 🔍 Motivation

AI and ML models rely on massive amounts of data — often sensitive and personal.  
While Differential Privacy provides a mathematical guarantee of privacy protection, its **effectiveness and trade-offs** are rarely evaluated in a consistent way.

Current challenges:
- No standardized method to compare DP algorithms across datasets or models  
- Trade-offs between **privacy, accuracy, fairness, and efficiency** often hidden  
- Lack of practical tools to visualize and quantify the impact of DP noise  

**PrivBench** solves this by providing a unified, reproducible benchmarking pipeline.

---

## 🧠 Framework Overview

PrivBench evaluates ML models along **four key dimensions**:

| Dimension | Description |
|------------|-------------|
| **Privacy Strength** | Quantifies the privacy budget (ε, δ) and privacy leakage risk using membership inference (MIA AUC). |
| **Model Utility** | Measures how model performance (accuracy, ROC-AUC) changes as privacy noise increases. |
| **Fairness** | Evaluates whether DP noise causes disparities between groups (DPD & EOD metrics). |
| **Efficiency** | Tracks computational cost (training time, model size) under different DP settings. |

---

## 📊 Example Results

**Dataset:** Wisconsin Breast Cancer Dataset  
**Model:** Logistic Regression trained with DP-SGD  
**Noise multipliers (σ):** 1.0, 5.0, 10.0  
**δ:** 1e-5  

| Setting | σ | ε (approx) | Accuracy | Fairness Gap (DPD/EOD) | Training Time (s) |
|----------|---|-------------|-----------|-------------------------|-------------------|
| Non-Private | – | ∞ | 0.98 | ~0.01 / 0.02 | 0.004 |
| DP-SGD | 1.0 | ~9.8 | 0.96 | ~0.02 / 0.03 | 0.007 |
| DP-SGD | 5.0 | ~2.0 | 0.94 | ~0.04 / 0.05 | 0.010 |
| DP-SGD | 10.0 | ~1.0 | 0.91 | ~0.06 / 0.07 | 0.012 |

🧾 *Higher σ → stronger privacy (lower ε) but reduced accuracy. Fairness gaps stay small.*

---

## 📈 Visualization

PrivBench automatically generates a **matplotlib plot** of accuracy vs. DP noise (σ), annotated with privacy budgets (ε).

<p align="center">
  <img src="acc_vs_sigma_with_baseline_avg.png" width="600" alt="Privacy–Utility Trade-off">
</p>

---

## Getting Started

1. Clone the repo

    ```bash
    git clone https://github.com/<yourusername>/PrivBench.git
    cd PrivBench
    ```

2. Set up your environment

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Run the benchmark

    ```bash
   python dp_benchmark_full_metrics_avg.py
    ```
4. View Ouputs

  Results CSV → dp_benchmark_full_metrics_avg.csv <br>
  Plot → acc_vs_sigma_with_baseline_avg.png


