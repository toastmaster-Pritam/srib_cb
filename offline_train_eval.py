#!/usr/bin/env python3
"""
offline_train_eval.py

Single-file offline pipeline for demo:
 - synthetic OBP bandit data
 - 10-day decayed short histories -> true long-term L_true
 - RF to predict L_true from context + first K days short features
 - learn alpha (scalar) and beta (3-vector) to form composite reward:
       r_comp = alpha * (short_features · beta) + (1-alpha) * L_hat
 - build LinUCB sufficient stats from logged data using r_comp
 - build per-action reward regressor r_hat(x,a) to compute expected-per-round rewards for policies
 - run OPE (IPS / SNIPS / DR) via obp
 - plot cumulative expected reward for uniform vs linucb
"""
import os
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import torch
from obp.dataset import SyntheticBanditDataset
from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting, SelfNormalizedInverseProbabilityWeighting, DoublyRobust

# -----------------------
# CONFIG (tweak for demo)
# -----------------------
SEED = 42
N_ROUNDS = 2000         # total synthetic logged rounds
N_ACTIONS = 10
DIM_CONTEXT = 12
DAYS = 10               # true long horizon
K_DAYS = 5              # partial days used for predicting L
DECAY = 0.95            # decay factor per day
RF_N_EST = 200
LINUCB_REG = 1.0
ALPHA_LIN = 0.1
RANDOM_STATE = SEED
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# 1) synthetic bandit data (context + logged action and pscore)
# -----------------------
print("[*] Generating synthetic OBP bandit data...")
ds = SyntheticBanditDataset(n_actions=N_ACTIONS, dim_context=DIM_CONTEXT, reward_type="continuous", random_state=SEED)
bf = ds.obtain_batch_bandit_feedback(n_rounds=N_ROUNDS)
# bf contains 'context', 'action', 'pscore' among others
X = bf["context"].astype(float)            # shape (N, DIM_CONTEXT)
A = bf["action"].astype(int)               # shape (N,)
P = bf.get("pscore", None)
if P is None:
    P = np.ones_like(A, dtype=float) / N_ACTIONS
else:
    P = np.array(P, dtype=float)

N = X.shape[0]
print(f"  n_rounds={N}, n_actions={N_ACTIONS}, dim={DIM_CONTEXT}")

# -----------------------
# 2) synthesize 10-day short histories and L_true (decayed cumulative)
# -----------------------
print("[*] Synthesizing 10-day short-term histories and true long-term (L_true)...")
rng = np.random.default_rng(SEED + 1)

# We'll create per-day short features: (click, revisit, watch_time) per day
SE_days = np.zeros((N, DAYS, 3), dtype=float)
for day in range(DAYS):
    # make probabilities depend a tiny bit on context first feature to create signal
    base_click_prob = 0.20 + 0.05 * np.tanh(X[:, 0] * 0.5)
    base_revisit_prob = 0.10 + 0.04 * np.tanh(X[:, 1] * 0.5)
    # randomness per day
    clicks = rng.binomial(1, np.clip(base_click_prob + rng.normal(0, 0.03, size=N), 0, 1))
    revisits = rng.binomial(1, np.clip(base_revisit_prob + rng.normal(0, 0.02, size=N), 0, 1))
    watch = np.clip(rng.normal(0.5 + 0.02 * (day / (DAYS-1)), 0.12, size=N), 0, 1)
    SE_days[:, day, 0] = clicks
    SE_days[:, day, 1] = revisits
    SE_days[:, day, 2] = watch

# daily score = weighted short engagement (these weights are internal design choices)
day_weights = np.array([0.35, 0.25, 0.40])  # click, revisit, watch
day_scores = (SE_days * day_weights.reshape(1, 1, 3)).sum(axis=2)  # shape (N, DAYS)

# decayed cumulative true long-term (L_true)
decay_powers = np.array([DECAY ** t for t in range(DAYS)])
L_true = (day_scores * decay_powers.reshape(1, DAYS)).sum(axis=1)   # shape (N,)
# normalize to [0,1]
L_true = (L_true - L_true.min()) / max(1e-9, (L_true.max() - L_true.min()))

print("  Synth done. L_true range:", float(L_true.min()), float(L_true.max()))

# short-term features we will use for composite reward:
# use aggregated first K_DAYS features (mean across K days)
SE_first_k = SE_days[:, :K_DAYS, :].mean(axis=1)   # shape (N, 3)

# -----------------------
# 3) Train RF to predict L_true from (context + first K days short features)
# -----------------------
print(f"[*] Training RF to predict L_true using context + first {K_DAYS} days short features...")
Xrf = np.hstack([X, SE_first_k])  # shape (N, DIM + 3)
Xrf_train, Xrf_test, y_train, y_test = train_test_split(Xrf, L_true, test_size=0.2, random_state=SEED)
rf = RandomForestRegressor(n_estimators=RF_N_EST, min_samples_leaf=3, n_jobs=-1, random_state=SEED)
rf.fit(Xrf_train, y_train)
r2 = rf.score(Xrf_test, y_test)
print("  RF trained. test R^2 (approx):", r2)

# predicted L_hat (for all rounds using available first-K-days)
L_hat = rf.predict(Xrf)   # numpy float64

# -----------------------
# 4) learn alpha and beta (composite params) via small torch fit
#    target: alpha * (SE_first_k @ beta) + (1-alpha) * L_hat  ~ L_true
# -----------------------
print("[*] Learning composite parameters alpha and beta via small gradient fit...")
# use a subset for stable fit (we have L_true for synthetic data so we can train on all)
mask = np.arange(N)
se_t = torch.tensor(SE_first_k[mask], dtype=torch.float32)           # (N,3)
lt_hat_t = torch.tensor(L_hat[mask].astype(np.float32), dtype=torch.float32)  # (N,)
lt_true_t = torch.tensor(L_true[mask].astype(np.float32), dtype=torch.float32) # (N,)

# initialize raw params
a_raw = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
b_raw = torch.tensor([0.34, 0.22, 0.44], requires_grad=True, dtype=torch.float32)
opt = torch.optim.Adam([a_raw, b_raw], lr=0.02)

for epoch in range(400):
    opt.zero_grad()
    alpha = torch.sigmoid(a_raw)                              # scalar in (0,1)
    beta = torch.nn.functional.softmax(b_raw, dim=0)          # 3-vector sum=1
    r_short = torch.matmul(se_t, beta)                        # (N,)
    r_comp = alpha * r_short + (1.0 - alpha) * lt_hat_t       # (N,)
    loss = torch.mean((r_comp - lt_true_t) ** 2) + 1e-4 * (alpha - 0.5) ** 2
    loss.backward()
    opt.step()
    # small early stop-ish (optional)
    if epoch % 100 == 0:
        pass

alpha_val = float(torch.sigmoid(a_raw).item())
beta_val = torch.softmax(b_raw.detach(), dim=0).numpy()
print("[*] learned alpha =", alpha_val, " beta =", beta_val)

# -----------------------
# 5) Composite reward R_comp for ALL rounds (backfill)
# -----------------------
print("[*] Building composite reward (backfill) for all rounds...")
r_short_all = SE_first_k @ beta_val   # numpy float64
R_comp = np.clip(alpha_val * r_short_all + (1 - alpha_val) * L_hat, 0.0, 1.0)   # (N,)

# -----------------------
# 6) Build LinUCB sufficient stats (A, b) from logged data (x, a, R_comp)
# -----------------------
print("[*] Building LinUCB sufficient stats (A, b) from synthetic data...")
d = DIM_CONTEXT
A_mats = np.stack([np.eye(d) * LINUCB_REG for _ in range(N_ACTIONS)])   # (n_actions, d, d)
b_vecs = np.zeros((N_ACTIONS, d), dtype=float)
counts = np.zeros(N_ACTIONS, dtype=int)
for x, a, r in zip(X, A, R_comp):
    A_mats[a] += np.outer(x, x)
    b_vecs[a] += r * x
    counts[a] += 1

# -----------------------
# 7) build candidate policy (linucb) action distribution per-round (n_rounds, n_actions)
# -----------------------
print("[*] Computing target policy distributions (uniform and linucb)...")
def linucb_action_dist_for(Amats, Bvecs, contexts, alpha_lin=ALPHA_LIN):
    n_rounds = contexts.shape[0]
    out = np.zeros((n_rounds, N_ACTIONS), dtype=float)
    for i, x in enumerate(contexts):
        xcol = x.reshape(d,1)
        sc = np.zeros(N_ACTIONS, dtype=float)
        for a in range(N_ACTIONS):
            Ainv = np.linalg.inv(Amats[a])
            theta = Ainv @ Bvecs[a].reshape(d,1)
            mean = float((theta.T @ xcol).item())
            std = float(np.sqrt((xcol.T @ Ainv @ xcol).item()))
            sc[a] = mean + alpha_lin * std
        ex = np.exp(sc - sc.max())
        probs = ex / ex.sum()
        out[i, :] = probs
    return out

ad_linucb = linucb_action_dist_for(A_mats, b_vecs, X, alpha_lin=ALPHA_LIN)
ad_uniform = np.ones_like(ad_linucb) / float(N_ACTIONS)   # (N, n_actions)

# -----------------------
# 8) Build bandit_feedback dict expected by obp OPE (2D action_dist — single slot)
# -----------------------
print("[*] Preparing bandit_feedback for OPE...")
# We need: n_rounds, n_actions, context, action, reward, pscore
bf_for_ope = {
    "n_rounds": int(N),
    "n_actions": int(N_ACTIONS),
    "context": X.astype(float),
    "action": A.astype(int),
    "reward": R_comp.astype(float),
    "pscore": np.where(np.isnan(P), 1.0 / N_ACTIONS, P).astype(float),
}

# -----------------------
# 9) Train per-action reward regressor r_hat(x,a) so we can compute per-round expected reward for plotting
#    We'll one-hot encode action and train RF to predict R_comp
# -----------------------
print("[*] Training per-action reward regressor (context + action_onehot -> R_comp) for plotting expected reward per round...")
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False, categories=[np.arange(N_ACTIONS)], handle_unknown='ignore')
A_onehot = enc.fit_transform(A.reshape(-1,1))  # (N, n_actions)
X_reg = np.hstack([X, A_onehot])               # (N, d + n_actions)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, R_comp, test_size=0.2, random_state=SEED)
rf_r = RandomForestRegressor(n_estimators=RF_N_EST, min_samples_leaf=3, n_jobs=-1, random_state=SEED + 1)
rf_r.fit(Xr_train, yr_train)
r2_r = rf_r.score(Xr_test, yr_test)
print("  r_hat reg trained. test R^2:", r2_r)

# predict r_hat for every (x,a) combination -> r_hat_matrix shape (N, n_actions)
r_hat_matrix = np.zeros((N, N_ACTIONS), dtype=float)
for i, x in enumerate(X):
    # build (n_actions, d + n_actions) input where onehot cycles
    action_oh = np.eye(N_ACTIONS, dtype=float)
    X_all = np.hstack([np.repeat(x.reshape(1, -1), N_ACTIONS, axis=0), action_oh])
    preds = rf_r.predict(X_all)   # (n_actions,)
    r_hat_matrix[i, :] = preds

# compute per-round expected reward under two policies
exp_reward_uniform = (ad_uniform * r_hat_matrix).sum(axis=1)
exp_reward_linucb = (ad_linucb * r_hat_matrix).sum(axis=1)

# cumulative expected reward
cum_uniform = np.cumsum(exp_reward_uniform)
cum_linucb = np.cumsum(exp_reward_linucb)

# -----------------------
# 10) Run Off-Policy Evaluation via obp
# -----------------------
print("[*] Running offline policy evaluation (IPS / SNIPS / DR)...")
ope = OffPolicyEvaluation(bandit_feedback=bf_for_ope,
                          ope_estimators=[InverseProbabilityWeighting(),
                                          SelfNormalizedInverseProbabilityWeighting(),
                                          DoublyRobust()])

res_uniform = ope.estimate_policy_values(action_dist=ad_uniform)
res_linucb = ope.estimate_policy_values(action_dist=ad_linucb)

def fmt_est(name, estd):
    return (f"{name:7s} | IPS={estd['ipw']:.4f} SNIPS={estd['snipw']:.4f} DR={estd['dr']:.4f} approx_true={estd.get('ground_truth', float('nan')):.4f}")

# OBP returns dicts; print basic numbers
print("  uniform | IPS={:.4f} SNIPS={:.4f} DR={:.4f}".format(float(res_uniform["ipw"]),
                                                            float(res_uniform["snipw"]),
                                                            float(res_uniform["dr"])))
print("  linucb  | IPS={:.4f} SNIPS={:.4f} DR={:.4f}".format(float(res_linucb["ipw"]),
                                                            float(res_linucb["snipw"]),
                                                            float(res_linucb["dr"])))
uplift_dr = float(res_linucb["dr"]) - float(res_uniform["dr"])
print("[*] Estimated uplift (linucb vs uniform) by DR = {:.6f}".format(uplift_dr))

# -----------------------
# 11) Plot cumulative expected reward (from r_hat_matrix approximation)
# -----------------------
print("[*] Saving cumulative expected reward plot...")
plt.figure(figsize=(8,4))
plt.plot(cum_uniform, label="uniform (expected via r_hat)", linewidth=2)
plt.plot(cum_linucb, label="linucb (expected via r_hat)", linewidth=2)
plt.xlabel("round (cumulative)")
plt.ylabel("cumulative expected reward (approx)")
plt.legend()
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "cum_reward_offline.png")
plt.savefig(out_path)
plt.close()
print("  saved:", out_path)

# -----------------------
# 12) Save summary JSON
# -----------------------
summary = {
    "n_rounds": int(N),
    "n_actions": int(N_ACTIONS),
    "dim_context": int(DIM_CONTEXT),
    "rf_r_test_r2": float(r2_r),
    "rf_Lhat_test_r2": float(r2),
    "alpha": float(alpha_val),
    "beta": beta_val.tolist(),
    "ope_uniform": {"ipw": float(res_uniform["ipw"]), "snipw": float(res_uniform["snipw"]), "dr": float(res_uniform["dr"])},
    "ope_linucb": {"ipw": float(res_linucb["ipw"]), "snipw": float(res_linucb["snipw"]), "dr": float(res_linucb["dr"])},
    "uplift_dr": float(uplift_dr),
    "timestamp": time.time(),
}

with open(os.path.join(OUTPUT_DIR, "summary_offline.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("[*] Summary saved to", os.path.join(OUTPUT_DIR, "summary_offline.json"))
print("[*] Done.")