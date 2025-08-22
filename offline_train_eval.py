#!/usr/bin/env python3
"""
offline_train_eval.py

Generates synthetic bandit logs, trains a long-term regressor, learns alpha/beta for composite reward,
trains a LinUCB (sufficient-stat) policy offline from composite rewards, and evaluates via OPE
(uses obp.ope).

Usage:
    python3 offline_train_eval.py --n_rounds 8000 --n_actions 10 --dim 12 --seed 42

Dependencies:
    pip install numpy torch scikit-learn joblib matplotlib pandas obp
"""
import os
import argparse
import numpy as np
import torch
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from obp.dataset import SyntheticBanditDataset
from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting, SelfNormalizedInverseProbabilityWeighting, DoublyRobust
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def gen_short_features(n, seed=0):
    rng = np.random.default_rng(seed)
    clicks = rng.binomial(1, 0.30, size=n)
    revis = rng.binomial(1, 0.20, size=n)
    watch = rng.random(n)  # in [0,1)
    return np.stack([clicks, revis, watch], axis=1)

def build_true_long_term(X, SE, rng, gamma=0.95):
    # For demo: true long-term is combination of short-term features + context signal + noise.
    # This matches the intuition: L depends on session behavior and context.
    ctx_sig = (X[:, 0] - X[:, 1])
    ctx_sig = ctx_sig / max(1.0, np.std(ctx_sig))
    # Weighted combination + noise
    L = 0.35 * SE[:, 2] + 0.25 * SE[:, 0] + 0.15 * SE[:, 1] + 0.25 * ctx_sig
    L = np.clip(L + rng.normal(0.0, 0.03, size=L.shape), 0.0, 1.0)
    # If you want a decayed-cumulative notion across days, you could simulate many days.
    return L

def learn_alpha_beta(SE, L_hat, L_true, n_steps=400, lr=0.02, device='cpu'):
    # Parametrization: alpha = sigmoid(a_raw), beta = softmax(b_raw)
    se_t = torch.tensor(SE, dtype=torch.float32, device=device)
    lt_hat_t = torch.tensor(L_hat, dtype=torch.float32, device=device)
    lt_true_t = torch.tensor(L_true, dtype=torch.float32, device=device)
    a_raw = torch.tensor(0.0, requires_grad=True, device=device)
    b_raw = torch.tensor([0.34, 0.22, 0.44], requires_grad=True, device=device)
    opt = torch.optim.Adam([a_raw, b_raw], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        alpha = torch.sigmoid(a_raw)
        beta = torch.nn.functional.softmax(b_raw, dim=0)
        r_short = se_t @ beta
        r_comp = alpha * r_short + (1.0 - alpha) * lt_hat_t
        loss = torch.mean((r_comp - lt_true_t) ** 2) + 1e-4 * (alpha - 0.5) ** 2
        loss.backward()
        opt.step()
    alpha_val = float(torch.sigmoid(a_raw).item())
    beta_val = torch.softmax(b_raw.detach(), dim=0).cpu().numpy()
    return alpha_val, beta_val

def build_linucb_stats(X, A, R, n_actions, d, reg_lambda=1.0):
    # A_mats shape: (n_actions, d, d), b_vecs: (n_actions, d)
    A_mats = np.stack([np.eye(d) * reg_lambda for _ in range(n_actions)])
    b_vecs = np.zeros((n_actions, d))
    counts = np.zeros(n_actions, dtype=int)
    for x, a, r in zip(X, A, R):
        A_mats[a] += np.outer(x, x)
        b_vecs[a] += r * x
        counts[a] += 1
    return A_mats, b_vecs, counts

def linucb_action_dist_from_stats(A_mats, b_vecs, alpha_lin, contexts):
    # contexts: (n_rounds, d)
    n_rounds = contexts.shape[0]
    n_actions = A_mats.shape[0]
    d = contexts.shape[1]
    ad = np.zeros((n_rounds, n_actions, 1))
    for i, x in enumerate(contexts):
        xcol = x.reshape(d, 1)
        sc = np.zeros(n_actions)
        for a in range(n_actions):
            Ainv = np.linalg.inv(A_mats[a])
            theta = Ainv @ b_vecs[a].reshape(d, 1)
            mean = float((theta.T @ xcol).item())
            std = float(np.sqrt((xcol.T @ Ainv @ xcol).item()))
            sc[a] = mean + alpha_lin * std
        ex = np.exp(sc - sc.max())
        p = ex / ex.sum()
        ad[i, :, 0] = p
    return ad

# -----------------------------
# Main: generate data, train, evaluate
# -----------------------------
def main(args):
    rng = np.random.default_rng(args.seed)

    print("[1] Generating synthetic OBP dataset...")
    ds = SyntheticBanditDataset(n_actions=args.n_actions, dim_context=args.dim, reward_type="binary", random_state=args.seed)
    bf = ds.obtain_batch_bandit_feedback(n_rounds=args.n_rounds)
    X = bf["context"].astype(float)            # (n_rounds, d)
    A = bf["action"].astype(int)               # (n_rounds,)
    P = bf["pscore"].astype(float)             # probability of chosen action under logging policy (per-round scalar)
    n_rounds = X.shape[0]
    d = X.shape[1]
    print(f"   rounds={n_rounds}, actions={args.n_actions}, dim={d}")

    print("[2] Simulating short-term engagement features and true long-term labels...")
    SE = gen_short_features(n_rounds, seed=args.seed + 1)  # (n_rounds, 3)
    L_true = build_true_long_term(X, SE, rng)
    # Train RF to predict L from context + SE
    print("[3] Training RandomForest regressor to predict L (L_hat)...")
    Xrf = np.hstack([X, SE])
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=args.seed, n_jobs=-1)
    rf.fit(Xrf, L_true)
    L_hat = rf.predict(Xrf)

    print("[4] Learning alpha & beta for composite reward (fit on full synthetic dataset)...")
    alpha_val, beta_val = learn_alpha_beta(SE, L_hat, L_true, n_steps=400, lr=0.02)
    print(f"   learned alpha={alpha_val:.4f}, beta={beta_val}")

    print("[5] Building composite reward and LinUCB stats from synthetic logs...")
    R_comp = np.clip(alpha_val * (SE @ beta_val) + (1 - alpha_val) * L_hat, 0.0, 1.0)

    A_mats, b_vecs, counts = build_linucb_stats(X, A, R_comp, args.n_actions, d, reg_lambda=args.reg_lambda)

    # Save artifacts
    os.makedirs("model_store", exist_ok=True)
    dump(rf, "model_store/rf_demo.joblib")
    np.savez("model_store/linucb_stats.npz", A_mats=A_mats, b_vecs=b_vecs, counts=counts, alpha=alpha_val, beta=beta_val)
    print("   saved rf + linucb stats to model_store/")

    print("[6] Offline evaluation (OPE) comparing uniform baseline vs candidate (LinUCB)")
    # Build bandit_feedback for OBP
    bf_ope = {
        "n_rounds": n_rounds,
        "context": X,
        "action": A,
        "reward": R_comp,
        "pscore": P
    }

    # Action dist for uniform baseline
    ad_uniform = np.ones((n_rounds, args.n_actions, 1)) / float(args.n_actions)

    # Action dist for candidate (LinUCB trained from composite rewards)
    ad_linucb = linucb_action_dist_from_stats(A_mats, b_vecs, args.alpha_lin, X)

    # run OPE
    ope = OffPolicyEvaluation(bandit_feedback=bf_ope,
                              ope_estimators=[InverseProbabilityWeighting(),
                                              SelfNormalizedInverseProbabilityWeighting(),
                                              DoublyRobust()])
    est_uniform = ope.estimate_policy_values(action_dist=ad_uniform)
    est_linucb = ope.estimate_policy_values(action_dist=ad_linucb)

    # Pretty print results (we extract available estimator names)
    def print_est(prefix, est_dict):
        print(f"--- {prefix} ---")
        for k, v in est_dict.items():
            try:
                val = float(v)
            except Exception:
                # sometimes returns arrays
                val = np.asarray(v).item() if np.asarray(v).size == 1 else v
            print(f"  {k}: {val}")
    print_est("Uniform (baseline)", est_uniform)
    print_est("LinUCB (candidate)", est_linucb)

    # Also compute simple empirical average reward on data for the rounds where logging action equals candidate action
    # (just to show an additional number)
    # compute argmax selection under LinUCB (greedy)
    scores = np.zeros((n_rounds, args.n_actions))
    for i, x in enumerate(X):
        xcol = x.reshape(d, 1)
        for a in range(args.n_actions):
            Ainv = np.linalg.inv(A_mats[a])
            theta = Ainv @ b_vecs[a].reshape(d, 1)
            mean = float((theta.T @ xcol).item())
            std = float(np.sqrt((xcol.T @ Ainv @ xcol).item()))
            scores[i, a] = mean + args.alpha_lin * std
    greedy_actions = np.argmax(scores, axis=1)
    mask = (greedy_actions == A)
    empirical_avg = float(np.nanmean(R_comp[mask])) if mask.sum() > 0 else float('nan')
    print(f"\nEmpirical avg composite reward on rounds where logging action == greedy LinUCB: {empirical_avg:.4f} (n={mask.sum()})")

    # Plot: OPE bar chart for available estimators (dr, ipw, snipw etc.)
    os.makedirs("monitor_outputs", exist_ok=True)
    keys = list(set(list(est_uniform.keys()) + list(est_linucb.keys())))
    labels = []
    uniform_vals = []
    linucb_vals = []
    for k in sorted(keys):
        labels.append(k)
        uniform_vals.append(float(est_uniform[k]) if k in est_uniform else np.nan)
        linucb_vals.append(float(est_linucb[k]) if k in est_linucb else np.nan)
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(6 + len(labels)*0.5, 4))
    plt.bar(x - width/2, uniform_vals, width, label='Uniform')
    plt.bar(x + width/2, linucb_vals, width, label='LinUCB')
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("OPE estimate (expected composite reward)")
    plt.title("Offline evaluation (OPE) estimates")
    plt.legend()
    plt.tight_layout()
    out = os.path.join("monitor_outputs", "ope_estimates.png")
    plt.savefig(out)
    print("Saved OPE plot to", out)

    # Done
    print("\nFinished offline train & evaluation. Artifacts in model_store/ and monitor_outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rounds", type=int, default=5000)
    parser.add_argument("--n_actions", type=int, default=10)
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--alpha_lin", type=float, default=0.2, help="LinUCB exploration multiplier for scoring")
    args = parser.parse_args()
    main(args)