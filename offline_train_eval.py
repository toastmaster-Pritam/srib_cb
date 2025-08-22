#!/usr/bin/env python3
"""
offline_train_eval.py

Generates synthetic bandit data (OBP), synthesizes 10-day short-term histories,
trains an RF to predict long-term engagement (L), learns composite reward
(alpha, beta), builds LinUCB stats, and evaluates policies offline using
IPS / SNIPS / Doubly Robust. Saves a cumulative-reward plot.

This script is demo-focused and self-contained.
"""
import os
import argparse
import json
from datetime import datetime
import numpy as np
import torch
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from obp.dataset import SyntheticBanditDataset
import matplotlib.pyplot as plt

# -------------------------
# Utilities / OPE functions
# -------------------------
def safe_pscore(pscore, n_actions, eps=1e-12):
    p = np.array(pscore, dtype=float)
    # replace nan or zeros with uniform fallback
    p[np.isnan(p)] = 1.0 / n_actions
    p[p < eps] = eps
    return p

def ips_estimate(r, a, pscore, pi_probs):
    n = len(r)
    idx = np.arange(n)
    w = pi_probs[idx, a] / pscore
    return float(np.mean(r * w))

def snips_estimate(r, a, pscore, pi_probs):
    n = len(r)
    idx = np.arange(n)
    w = pi_probs[idx, a] / pscore
    denom = np.sum(w)
    if denom == 0:
        return 0.0
    return float(np.sum(r * w) / denom)

def dr_estimate(r, a, pscore, pi_probs, q_hat):
    # q_hat: (n_rounds, n_actions) ndarray
    n = len(r)
    idx = np.arange(n)
    q_pi = np.sum(pi_probs * q_hat, axis=1)          # E_pi[q_hat(s,.)]
    w = pi_probs[idx, a] / pscore
    # q_hat at chosen actions
    q_chosen = q_hat[idx, a]
    dr_vals = q_pi + w * (r - q_chosen)
    return float(np.mean(dr_vals))

# -------------------------
# LinUCB scoring util
# -------------------------
def linucb_action_scores(A_dict, b_dict, x, alpha_lin):
    # A_dict, b_dict: dict[action] -> ndarray
    d = x.shape[0]
    xcol = x.reshape(d,1)
    n_actions = len(A_dict)
    scores = np.zeros(n_actions)
    for a in range(n_actions):
        A = A_dict[a]; b = b_dict[a]
        Ainv = np.linalg.inv(A)
        theta = Ainv @ b.reshape(d,1)
        mean = float((theta.T @ xcol).item())
        std = float(np.sqrt((xcol.T @ Ainv @ xcol).item()))
        scores[a] = mean + alpha_lin * std
    return scores

# -------------------------
# Main pipeline
# -------------------------
def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)

    print("[*] Generating synthetic OBP bandit data...")
    ds = SyntheticBanditDataset(n_actions=args.n_actions, dim_context=args.dim, reward_type="binary", random_state=args.seed)
    bf = ds.obtain_batch_bandit_feedback(n_rounds=args.n_rounds)
    X = np.array(bf["context"], dtype=float)     # (n_rounds, dim)
    A = np.array(bf["action"], dtype=int)        # (n_rounds,)
    P = np.array(bf["pscore"], dtype=float)     # behavior propensities

    n = X.shape[0]
    print(f"  n_rounds={n}, n_actions={args.n_actions}, dim={args.dim}")

    # -------------------------
    # Synthesize 10-day short histories and compute true L
    # -------------------------
    print("[*] Synthesizing 10-day short-term histories and true long-term (L)...")
    rng = np.random.default_rng(args.seed)
    days = args.n_days
    decay = args.decay

    # weights for per-day short engagement (click, revisit, watch_time)
    w_click, w_revisit, w_watch = 0.3, 0.2, 0.5

    # For each round create days x 3 short features and compute per-day short score
    shorts_all = rng.binomial(1, 0.30, size=(n, days, 1))          # clicks ~ Bernoulli
    revis_all = rng.binomial(1, 0.20, size=(n, days, 1))          # revisits
    watch_all = rng.random((n, days, 1))                          # watch time in [0,1)
    # per-day short vector (n, days, 3)
    day_feats = np.concatenate([shorts_all, revis_all, watch_all], axis=2)

    # per-day scalar short score
    day_scores = (w_click * day_feats[:,:,0] + w_revisit * day_feats[:,:,1] + w_watch * day_feats[:,:,2])
    # decayed cumulative long-term: L_true = sum_{d=1..days} decay^(d-1) * day_scores[:,d-1]
    powers = np.array([decay**i for i in range(days)])
    L_true = np.clip((day_scores * powers[None,:]).sum(axis=1), 0.0, 1.0)   # shape (n,)

    # Observed short events available at decision time: use day 1 features (index 0)
    SE_day1 = day_feats[:,0,:]   # shape (n,3)

    # -------------------------
    # Prepare training data for RF: predict L_true from context + partial first_k days
    # -------------------------
    first_k = args.partial_days
    print(f"[*] Training RF to predict L_true using context + first {first_k} days short features...")
    feats = []
    for i in range(n):
        # flatten first_k days of day_feats
        flat = day_feats[i, :first_k, :].reshape(-1)
        feats.append(np.concatenate([X[i], flat]))
    feats = np.vstack(feats)
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(feats, L_true, test_size=0.2, random_state=args.seed)
    rf = RandomForestRegressor(n_estimators=args.n_estimators, min_samples_leaf=3, random_state=args.seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("  RF trained. test R^2 (approx):", rf.score(X_test, y_test))

    # Save RF for demo inspection
    dump(rf, os.path.join(outdir, "rf_Lhat.joblib"))

    # compute L_hat for all rounds using only context + first_k days (as in online)
    L_hat = rf.predict(feats)   # shape (n,)

    # -------------------------
    # Learn alpha and beta (combining short-term summary + L_hat to predict L_true)
    # r_short will be a weighted sum of SE_day1 using beta (softmax)
    # -------------------------
    print("[*] Learning composite parameters alpha and beta via small gradient fit...")
    se_t = torch.tensor(SE_day1, dtype=torch.float32)           # (n,3)
    lt_hat_t = torch.tensor(L_hat, dtype=torch.float32)
    lt_true_t = torch.tensor(L_true, dtype=torch.float32)

    a_raw = torch.tensor(0.0, requires_grad=True)
    b_raw = torch.tensor([0.34,0.22,0.44], requires_grad=True)   # initial short weights
    opt = torch.optim.Adam([a_raw, b_raw], lr=0.02)
    for _ in range(400):
        opt.zero_grad()
        alpha = torch.sigmoid(a_raw)
        beta = torch.nn.functional.softmax(b_raw, dim=0)
        r_short = se_t @ beta
        r_comp = alpha * r_short + (1.0 - alpha) * lt_hat_t
        loss = torch.mean((r_comp - lt_true_t)**2) + 1e-4 * (alpha - 0.5)**2
        loss.backward(); opt.step()
    alpha_val = float(torch.sigmoid(a_raw).item())
    beta_val = torch.softmax(b_raw.detach(), dim=0).numpy()
    print("  learned alpha =", alpha_val, "beta =", beta_val)

    # -------------------------
    # Composite reward for all rounds (this is what we'll treat as the 'observed' reward for training LinUCB)
    # -------------------------
    R_comp = np.clip(alpha_val * (SE_day1 @ beta_val) + (1.0 - alpha_val) * L_hat, 0.0, 1.0)

    # -------------------------
    # Build LinUCB A & b using contexts X and composite reward R_comp
    # -------------------------
    print("[*] Building LinUCB sufficient stats (A, b) from synthetic data...")
    d = X.shape[1]
    n_actions = args.n_actions
    A_mats = np.stack([np.eye(d) * args.reg_lambda for _ in range(n_actions)])
    b_vecs = np.zeros((n_actions, d), dtype=float)
    counts = np.zeros(n_actions, dtype=int)
    for x,a,r in zip(X, A, R_comp):
        A_mats[a] += np.outer(x, x)
        b_vecs[a] += r * x
        counts[a] += 1

    # Save A,b for inspection
    np.savez(os.path.join(outdir, "linucb_stats.npz"), A=A_mats, b=b_vecs, counts=counts)

    # -------------------------
    # Define target policies (uniform, LinUCB-softmax) and compute pi_probs matrix
    # -------------------------
    print("[*] Computing target policy distributions...")
    # For uniform: each round same uniform probs
    pi_uniform = np.ones((n, n_actions), dtype=float) / n_actions

    # For LinUCB: compute scores then softmax per-round
    alpha_lin = args.alpha_lin
    pi_linucb = np.zeros((n, n_actions), dtype=float)
    A_dict = {a: A_mats[a] for a in range(n_actions)}
    b_dict = {a: b_vecs[a] for a in range(n_actions)}
    for i in range(n):
        scores = linucb_action_scores(A_dict, b_dict, X[i], alpha_lin)
        # softmax to produce stochastic policy
        ex = np.exp(scores - np.max(scores))
        pi_linucb[i] = ex / ex.sum()

    # -------------------------
    # Prepare inputs for OPE
    # -------------------------
    behavior_pscore = safe_pscore(P, n_actions)
    # Observed action indices (A), and observed rewards r (we use R_comp as the 'observed' composite reward)
    r_obs = R_comp.copy()
    a_obs = A.copy()

    # q_hat matrix for DR: we will use the composite predicted (alpha*(SE@beta)+(1-alpha)*L_hat)
    # This is already R_comp (per-round predicted), but DR expects shape (n_rounds, n_actions)
    q_hat = np.tile(R_comp.reshape(n,1), (1, n_actions))

    # compute OPE metrics for both policies
    print("[*] Running offline policy evaluation (IPS / SNIPS / DR)...")
    results = {}
    for name, pi in [("uniform", pi_uniform), ("linucb", pi_linucb)]:
        ips = ips_estimate(r_obs, a_obs, behavior_pscore, pi)
        snips = snips_estimate(r_obs, a_obs, behavior_pscore, pi)
        dr = dr_estimate(r_obs, a_obs, behavior_pscore, pi, q_hat)
        # approximate true expected reward under target (because reward is independent of action in our sim, the ground truth is simply mean(R_comp))
        true_ev = float(np.mean(np.sum(pi * q_hat, axis=1)))
        results[name] = {"ips": ips, "snips": snips, "dr": dr, "approx_true": true_ev}
        print(f"  {name:7s} | IPS={ips:.4f} SNIPS={snips:.4f} DR={dr:.4f} approx_true={true_ev:.4f}")

    # compute uplift between linucb and uniform (DR)
    uplift_dr = results["linucb"]["dr"] - results["uniform"]["dr"]
    print(f"[*] Estimated uplift (linucb vs uniform) by DR = {uplift_dr:.6f}")

    # -------------------------
    # Plot cumulative expected reward over rounds (using q_pi per-round)
    # -------------------------
    print("[*] Saving cumulative expected reward plot...")
    q_uniform = np.sum(pi_uniform * q_hat, axis=1)
    q_linucb = np.sum(pi_linucb * q_hat, axis=1)
    t = np.arange(1, n+1)
    cum_u = np.cumsum(q_uniform)
    cum_l = np.cumsum(q_linucb)
    plt.figure(figsize=(8,4))
    plt.plot(t, cum_u, label="uniform")
    plt.plot(t, cum_l, label="linucb")
    plt.legend()
    plt.title("Cumulative expected reward (offline)")
    plt.tight_layout()
    outplot = os.path.join(outdir, "cum_reward_offline.png")
    plt.savefig(outplot)
    plt.close()
    print("  saved:", outplot)

    # -------------------------
    # Save a small JSON summary for easy inspection
    # -------------------------
    summary = {
        "n_rounds": int(n),
        "n_actions": int(n_actions),
        "alpha_learned": float(alpha_val),
        "beta_learned": list(map(float, beta_val.tolist())),
        "linucb_alpha": float(alpha_lin),
        "ope_results": results,
        "uplift_dr": float(uplift_dr),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    with open(os.path.join(outdir, "summary_offline.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[*] Summary saved to outputs/summary_offline.json")
    print("[*] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rounds", type=int, default=2000)
    parser.add_argument("--n_actions", type=int, default=10)
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=150)
    parser.add_argument("--partial_days", type=int, default=3,
                        help="how many initial days available to the regressor")
    parser.add_argument("--n_days", type=int, default=10, help="ground-truth horizon (days)")
    parser.add_argument("--decay", type=float, default=0.95, help="decay per day for long-term sum")
    parser.add_argument("--alpha_lin", type=float, default=0.1, help="LinUCB exploration multiplier")
    parser.add_argument("--reg_lambda", type=float, default=1.0, help="LinUCB regularization")
    args = parser.parse_args()
    main(args)