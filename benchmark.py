# DP Benchmark (mean over seeds) + Plot with Non-private baseline
# Sigma values: 1.0, 5.0, 10.0
# Epsilon labels BELOW the dots + faint y=0.90 gridline
# Saves CSV and PNG with rounded metrics

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# ----------------- Helpers -----------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logistic_loss_and_gradients(X, y, w, b, l2=1e-4):
    z = X.dot(w) + b
    p = sigmoid(z)
    eps = 1e-12
    loss = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)) + 0.5*l2*np.sum(w*w)
    err = (p - y)
    gw_per = err[:, None] * X + l2 * w
    gb_per = err
    return loss, gw_per, gb_per

def clip_per_example(grads, C):
    norms = np.linalg.norm(grads, axis=1) + 1e-12
    factors = np.minimum(1.0, C / norms)
    return grads * factors[:, None], factors

def dp_sgd_logreg(X, y, *, epochs=30, batch_size=128, lr=0.18, l2=1e-4, C=1.0, sigma=1.0, seed=42):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    idx = np.arange(n)
    steps = 0
    t0 = time.time()
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            Xb, yb = X[idx[s:e]], y[idx[s:e]]
            _, gw_per, gb_per = logistic_loss_and_gradients(Xb, yb, w, b, l2=l2)
            gw_clip, _ = clip_per_example(gw_per, C)
            gb_clip, _ = clip_per_example(gb_per.reshape(-1,1), C); gb_clip = gb_clip.ravel()
            gw_sum = gw_clip.sum(axis=0)
            gb_sum = gb_clip.sum()
            gw_noisy = gw_sum + rng.normal(0.0, sigma*C, size=gw_sum.shape)
            gb_noisy = gb_sum + rng.normal(0.0, sigma*C, size=())
            m = max(1, e - s)
            w -= lr * (gw_noisy / m)
            b -= lr * (gb_noisy / m)
            steps += 1
    train_time = time.time() - t0
    return {"w": w, "b": b, "train_time": train_time, "steps": steps}

def predict_proba(model, X, sklearn_model=False):
    if sklearn_model:
        return model.predict_proba(X)[:, 1]
    return sigmoid(X.dot(model["w"]) + model["b"])

def model_size_kb(model, sklearn_model=False):
    params = (model.coef_.size + model.intercept_.size) if sklearn_model else (model["w"].size + 1)
    return (params * 8) / 1024.0

# Privacy accounting (conservative RDP, no subsampling)
def rdp_gaussian_per_step(alpha, sigma):
    return alpha / (2.0 * sigma**2)

def rdp_to_eps(delta, rdp, alpha):
    return rdp + np.log(1.0 / delta) / (alpha - 1.0)

def approx_eps(delta, sigma, steps, alphas=range(2, 65)):
    return float(min(rdp_to_eps(delta, steps * rdp_gaussian_per_step(a, sigma), a) for a in alphas))

# Membership Inference AUC (loss-based)
def mia_auc(model, X_train, y_train, X_test, y_test, sklearn_model=False):
    def per_sample_loss(X, y):
        p = predict_proba(model, X, sklearn_model=sklearn_model); eps = 1e-12
        return -(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
    lt = per_sample_loss(X_train, y_train)
    l0 = per_sample_loss(X_test, y_test)
    scores = np.concatenate([-lt, -l0])
    labels = np.concatenate([np.ones_like(lt), np.zeros_like(l0)])
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, scores)

# Fairness metrics
def demographic_parity_difference(y_pred, A):
    p0 = y_pred[A==0].mean() if np.any(A==0) else 0.0
    p1 = y_pred[A==1].mean() if np.any(A==1) else 0.0
    return float(p0 - p1)

def equal_opportunity_difference(y_true, y_pred, A):
    def tpr(y, yhat):
        pos = np.sum(y==1)
        return (np.sum((y==1) & (yhat==1)) / pos) if pos>0 else 0.0
    return float(tpr(y_true[A==0], y_pred[A==0]) - tpr(y_true[A==1], y_pred[A==1]))

# ----------------- Data -----------------
data = load_breast_cancer()
X_raw, y = data.data, data.target
X = StandardScaler().fit_transform(X_raw)

# Pseudo protected attribute
A_attr = (X_raw[:, 1] > np.median(X_raw[:, 1])).astype(int)

X_tr, X_te, y_tr, y_te, A_tr, A_te = train_test_split(
    X, y, A_attr, test_size=0.30, random_state=7, stratify=y
)

# ----------------- Non-private baseline -----------------
t0 = time.time()
baseline = LogisticRegression(max_iter=1000, solver='lbfgs')
baseline.fit(X_tr, y_tr)
t_nonpriv = time.time() - t0

yprob_np = baseline.predict_proba(X_te)[:, 1]
ypred_np = (yprob_np >= 0.5).astype(int)
acc_np = accuracy_score(y_te, ypred_np)
auc_np = roc_auc_score(y_te, yprob_np)
mia_np = mia_auc(baseline, X_tr, y_tr, X_te, y_te, sklearn_model=True)
dpd_np = demographic_parity_difference(ypred_np, A_te)
eod_np = equal_opportunity_difference(y_te, ypred_np, A_te)
mem_np = model_size_kb(baseline, sklearn_model=True)

# ----------------- DP-SGD variants (σ = 1.0, 5.0, 10.0), averaged over seeds -----------------
delta = 1e-5
sigmas = [5.0, 10.0, 15.0]
seeds = [11, 23, 37, 49, 61]  # multiple runs

def eval_sigma(sigma):
    accs, aucs, mias, dpds, eods, times = [], [], [], [], [], []
    steps_val = None
    for sd in seeds:
        model = dp_sgd_logreg(
            X_tr, y_tr,
            epochs=30, batch_size=128, lr=0.18, l2=1e-4,
            C=1.0, sigma=sigma, seed=sd
        )
        yprob = predict_proba(model, X_te, sklearn_model=False)
        ypred = (yprob >= 0.5).astype(int)
        accs.append(accuracy_score(y_te, ypred))
        aucs.append(roc_auc_score(y_te, yprob))
        mias.append(mia_auc(model, X_tr, y_tr, X_te, y_te, sklearn_model=False))
        dpds.append(demographic_parity_difference(ypred, A_te))
        eods.append(equal_opportunity_difference(y_te, ypred, A_te))
        times.append(model["train_time"])
        steps_val = model["steps"]
    eps = approx_eps(delta=delta, sigma=sigma, steps=steps_val)
    return {
        "Sigma": sigma,
        "Accuracy": np.mean(accs),
        "ROC_AUC": np.mean(aucs),
        "MIA_AUC (higher=worse privacy)": np.mean(mias),
        "Fairness: DPD": np.mean(dpds),
        "Fairness: EOD": np.mean(eods),
        "Train Time (s)": np.mean(times),
        "Model Size (KB)": model_size_kb({"w": np.zeros(X_tr.shape[1])}, sklearn_model=False),
        "DP ε (approx)": eps,
        "DP δ": delta,
        "Runs": len(seeds)
    }

rows = [{
    "Model": "Non-Private LR",
    "Sigma": np.nan,
    "Accuracy": acc_np,
    "ROC_AUC": auc_np,
    "MIA_AUC (higher=worse privacy)": mia_np,
    "Fairness: DPD": dpd_np,
    "Fairness: EOD": eod_np,
    "Train Time (s)": t_nonpriv,
    "Model Size (KB)": mem_np,
    "DP ε (approx)": np.nan,
    "DP δ": np.nan,
    "Runs": 1
}]

for sigma in sigmas:
    res = eval_sigma(sigma)
    res["Model"] = f"DP-SGD LR (σ={sigma})"
    rows.append(res)

df = pd.DataFrame(rows)

# Round all numeric columns to 4 decimals
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].astype(float).round(4)

print(df)

# ----------------- Plot -----------------
plt.figure(figsize=(7.6, 4.8))
baseline_acc = float(df.loc[df["Model"]=="Non-Private LR", "Accuracy"])
plt.axhline(baseline_acc, linestyle='--', linewidth=1.6,
            label=f'Non-Private Accuracy')

dp_only = df.dropna(subset=["Sigma"]).copy()
x_labels = [f"σ={s:g}" for s in dp_only["Sigma"]]
x_pos = np.arange(len(x_labels))

y_acc = dp_only["Accuracy"].astype(float).values
eps_vals = dp_only["DP ε (approx)"].astype(float).values

plt.scatter(x_pos, y_acc, s=95, label="DP-SGD")

# Annotate ε BELOW each point
for i, (x, yv, e) in enumerate(zip(x_pos, y_acc, eps_vals)):
    plt.annotate(f"ε≈{e:.1f}", (x, yv), xytext=(0, -12),
                 textcoords="offset points", ha="center", va="top", fontsize=9)

# Faint horizontal gridline at 0.90
# plt.axhline(0.90, color="gray", linestyle=":", linewidth=1, alpha=0.5)

plt.xticks(x_pos, x_labels)
plt.ylim(0.75, 1.01)
plt.xlabel("DP Setting (Noise Multiplier σ)")
plt.ylabel("Accuracy (test; mean over seeds)")
plt.title("Accuracy vs DP-SGD Noise (σ)\nNon-Private Baseline")
plt.legend()
plt.tight_layout()

# Save artifacts
df.to_csv("dp_benchmark_full_metrics_avg.csv", index=False)
plt.savefig("acc_vs_sigma_with_baseline_avg.png", dpi=200)
plt.show()

print("\nSaved:")
print("- Table: dp_benchmark_full_metrics_avg.csv")
print("- Plot : acc_vs_sigma_with_baseline_avg.png")
