# offline_demo_strong_uplift.py
"""
Self-contained offline demo that produces a clear uplift for a context-aware LinUCB over Uniform.
Run:
    python3 offline_demo_strong_uplift.py
Requires:
    numpy, sklearn, matplotlib, joblib, torch, obp
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import torch
from obp.ope import OffPolicyEvaluation, InverseProbabilityWeighting, SelfNormalizedInverseProbabilityWeighting, DoublyRobust
from obp.dataset import SyntheticBanditDataset
from datetime import datetime, timezone

np.random.seed(42)
torch.manual_seed(42)

OUTDIR = "outputs_strong_demo"
os.makedirs(OUTDIR, exist_ok=True)

# --- DEMO CONFIG: tune these to make uplift larger/smaller ---
N_ROUNDS = 5000         # more data -> easier to learn
N_ACTIONS = 10
DIM = 12
SHORT_DAYS = 3          # short-term features length
NOISE = 0.03
RF_ESTIMATORS = 300
LINUCb_ALPHA = 0.2      # exploration factor when computing LinUCB scores for target policy
# -----------------------------------------------------------------

def sigmoid(x): return 1/(1+np.exp(-x))

def generate_strong_contextual_data(n_rounds, n_actions, dim, short_days, seed=42):
    """
    Generate:
      - context X (n_rounds x dim)
      - chosen action A (n_rounds) (from OBP's synthetic logging policy)
      - logging propensities P (n_rounds)
      - short-term per-round vector SE (n_rounds x short_days)
      - true long-term L_true that *depends on action and context*
    Key idea: each action has its own weight vector; best action depends on context.
    """
    rng = np.random.default_rng(seed)
    ds = SyntheticBanditDataset(n_actions=n_actions, dim_context=dim, reward_type="binary", random_state=seed)
    bf = ds.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    X = bf["context"]          # shape (n_rounds, dim)
    A = bf["action"]
    P = bf["pscore"]

    # short-term features (day-level interactions) - small noise + dependence on action for realism
    clicks = rng.binomial(1, 0.25, size=n_rounds)
    revis = rng.binomial(1, 0.18, size=n_rounds)
    watch = np.clip(rng.normal(0.45, 0.15, size=n_rounds), 0., 1.)
    SE = np.stack([clicks, revis, watch], axis=1)  # shape (n_rounds, 3)

    # Create arm-specific weight vectors so optimal action depends on context
    arm_weights = rng.normal(0, 1.0, size=(n_actions, dim))
    # amplify structure so linear model strong: multiply some arms by factor
    for i in range(n_actions):
        arm_weights[i] *= (1.0 + 0.2 * (i % 3))

    # Make true long-term L_true depend on context·arm_weights[action] + short-term decayed sum
    # Also add a nonlinearity (sigmoid) to make reward in [0,1].
    ctx_effect = np.array([ (X[i] @ arm_weights[A[i]]) for i in range(n_rounds) ])
    # scale to reasonable range
    ctx_effect = (ctx_effect - ctx_effect.mean()) / max(1e-6, ctx_effect.std())
    # Short-term cumulative with decay; weight for short-term larger so composite is meaningful
    decay = np.array([0.6, 0.3, 0.1])  # decayed days -> strong immediate influence
    se_score = (SE * decay).sum(axis=1)
    # Build L_true: sigmoid of linear mixture
    L_true = sigmoid(1.2 * ctx_effect + 1.6 * se_score + rng.normal(0, NOISE, size=n_rounds))
    # Clip to [0,1]
    L_true = np.clip(L_true, 0.0, 1.0)

    return {
        "context": X, "action": A, "pscore": P, "short": SE, "long_true": L_true, "arm_weights": arm_weights
    }

def fit_rf_predictor(X_ctx, SE, L_true):
    """
    Train RF regressor to predict L_true from context + short features.
    Returns fitted rf and predictions for full X.
    """
    Xrf = np.hstack([X_ctx, SE])
    rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, min_samples_leaf=3, n_jobs=-1, random_state=42)
    rf.fit(Xrf, L_true)
    preds = rf.predict(Xrf)
    return rf, preds



def learn_alpha_beta(SE_train, Lhat_train, Ltrue_train, epochs=400, lr=0.02, device=None):
    """
    Fit scalar alpha in (0,1) and softmax(beta) weights over short features to minimize MSE:
      r_comp = alpha * (SE @ beta) + (1-alpha) * Lhat
    Returns (alpha_float, beta_numpy_array)
    Ensures all tensors are float32 to avoid dtype mismatch errors.
    """
    # ensure numpy arrays and float32
    SE_train = np.asarray(SE_train, dtype=np.float32)
    Lhat_train = np.asarray(Lhat_train, dtype=np.float32)
    Ltrue_train = np.asarray(Ltrue_train, dtype=np.float32)

    # device (cpu is fine for demo)
    if device is None:
        device = torch.device("cpu")

    se_t = torch.from_numpy(SE_train).to(dtype=torch.float32, device=device)       # (n_samples, n_short)
    lt_hat_t = torch.from_numpy(Lhat_train).to(dtype=torch.float32, device=device) # (n_samples,)
    lt_t = torch.from_numpy(Ltrue_train).to(dtype=torch.float32, device=device)    # (n_samples,)

    # initialize learnable params as float32
    a_raw = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    b_raw = torch.tensor(np.ones(SE_train.shape[1], dtype=np.float32), dtype=torch.float32,
                         requires_grad=True, device=device)

    opt = torch.optim.Adam([a_raw, b_raw], lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        alpha = torch.sigmoid(a_raw)                           # scalar float32
        beta = torch.nn.functional.softmax(b_raw, dim=0)       # (n_short,) float32

        # r_short: (n_samples,) = SE (n x k) @ beta (k,)
        r_short = se_t.matmul(beta)

        # composite reward per-sample
        r_comp = alpha * r_short + (1.0 - alpha) * lt_hat_t

        loss = torch.mean((r_comp - lt_t) ** 2)
        loss.backward()
        opt.step()

    alpha_val = float(torch.sigmoid(a_raw).item())
    beta_val = torch.softmax(b_raw.detach(), dim=0).cpu().numpy().astype(float)
    return alpha_val, beta_val

def build_linucb_stats(X, A, R_comp, n_actions, d, reg=1.0):
    A_mats = np.stack([np.eye(d) * reg for _ in range(n_actions)])
    b_vecs = np.zeros((n_actions, d))
    counts = np.zeros(n_actions, dtype=int)
    for x, a, r in zip(X, A, R_comp):
        # r expected to be scalar in [0,1]
        A_mats[a] += np.outer(x, x)
        b_vecs[a] += r * x
        counts[a] += 1
    return A_mats, b_vecs, counts

def linucb_action_dist_from_stats(A_mats, b_vecs, alpha_lin, X_ctx):
    """
    For each context, compute LinUCB score for each action and return action-distributions (softmax over scores).
    Returns array of shape (n_rounds, n_actions).
    """
    n_actions = len(A_mats)
    n_rounds = X_ctx.shape[0]
    ad = np.zeros((n_rounds, n_actions))
    d = X_ctx.shape[1]
    for i, x in enumerate(X_ctx):
        xcol = x.reshape(d, 1)
        sc = np.zeros(n_actions)
        for a in range(n_actions):
            Ainv = np.linalg.inv(A_mats[a])
            theta = Ainv @ b_vecs[a].reshape(d, 1)
            mean = float((theta.T @ xcol).item())
            std = float(np.sqrt((xcol.T @ Ainv @ xcol).item()))
            sc[a] = mean + alpha_lin * std
        ex = np.exp(sc - sc.max())
        ad[i, :] = ex / ex.sum()
    return ad

def uniform_action_dist(n_rounds, n_actions):
    return np.ones((n_rounds, n_actions)) / float(n_actions)

def run_ope_and_report(bandit_feedback, ad_uniform, ad_linucb):
    # bandit_feedback is a dict with keys: n_rounds, context, action, reward, pscore
    ope = OffPolicyEvaluation(bandit_feedback=bandit_feedback,
                              ope_estimators=[InverseProbabilityWeighting(),
                                              SelfNormalizedInverseProbabilityWeighting(),
                                              DoublyRobust()])
    est_uniform = ope.estimate_policy_values(action_dist=ad_uniform)
    est_linucb = ope.estimate_policy_values(action_dist=ad_linucb)

    def summarise(name, est):
        ips = float(est["ipw"])
        snip = float(est["snip"])
        dr = float(est["dr"])
        return f"{name:7s} | IPS={ips:.4f} SNIPS={snip:.4f} DR={dr:.4f}"

    print("Running OPE for Uniform and LinUCB...")
    print("  " + summarise("uniform", est_uniform))
    print("  " + summarise("linucb", est_linucb))
    uplift = float(est_linucb["dr"] - est_uniform["dr"])
    print(f"Estimated uplift (linucb - uniform) by DR = {uplift:.6f}")
    return est_uniform, est_linucb, uplift

def plot_cumulative_expected(bandit_feedback, ad_uniform, ad_linucb, out_path):
    # cumulative expected reward: for each round t, expected reward under policy is sum_a pi_t(a|x_t)*expected_reward(a|x_t)
    # we approximate expected_reward(a|x) by predicted composite reward r_comp computed earlier in bandit_feedback["reward"]
    R = np.array(bandit_feedback["reward"])  # reward per logged event (we will use predicted composite)
    # But we need expected reward for each action at each context - we will reconstruct using LinUCB theta from stats is easier.
    # For simplicity: compute expected instantaneous value for each logged context as average of policy's distribution dot logged reward.
    # This is an approximation but fine for demo cumulative curves.
    n_rounds = bandit_feedback["n_rounds"]
    # expected instant reward for a policy on logged rounds: sum_a pi(a|x_t) * r_hat_for_a(x_t)
    # We'll estimate r_hat_for_a(x_t) using logged reward if action==a else use average reward per action.
    A = np.array(bandit_feedback["action"])
    # average reward per action (fallback)
    avg_per_action = {}
    for a in np.unique(A):
        mask = (A == a)
        avg_per_action[a] = float(np.nanmean(R[mask])) if mask.sum() else 0.0
    # build r_est matrix
    r_est = np.zeros((n_rounds, bandit_feedback["n_actions"]))
    for t in range(n_rounds):
        for a in range(bandit_feedback["n_actions"]):
            if A[t] == a and not np.isnan(R[t]):
                r_est[t, a] = R[t]
            else:
                r_est[t, a] = avg_per_action.get(a, 0.0)
    exp_uniform = (ad_uniform * r_est).sum(axis=1)
    exp_linucb = (ad_linucb * r_est).sum(axis=1)
    cum_u = np.cumsum(exp_uniform)
    cum_l = np.cumsum(exp_linucb)
    plt.figure(figsize=(8,4))
    plt.plot(cum_u, label="uniform (expected cum reward)")
    plt.plot(cum_l, label="linucb (expected cum reward)")
    plt.legend()
    plt.title("Cumulative expected reward (approx)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved cumulative expected reward plot to", out_path)

def main():
    print("[*] Generating synthetic OBP bandit data...")
    gen = generate_strong_contextual_data(N_ROUNDS, N_ACTIONS, DIM, SHORT_DAYS, seed=42)
    X = gen["context"]; A = gen["action"]; P = gen["pscore"]; SE = gen["short"]; L_true = gen["long_true"]

    print("[*] Training RF to predict L_true from context + short features...")
    rf, Lhat = fit_rf_predictor(X, SE, L_true)
    # quick quality check
    from sklearn.metrics import r2_score
    r2 = r2_score(L_true, Lhat)
    print(f"  RF trained. test R^2 (approx): {r2:.4f}")

    # learn alpha and beta on a subset (use all here)
    print("[*] Learning composite parameters alpha and beta via small gradient fit...")
    alpha, beta = learn_alpha_beta(SE, Lhat, L_true, epochs=400, lr=0.02)
    print("  learned alpha =", alpha, "beta =", beta)

    print("[*] Building composite reward r_comp = alpha * (SE @ beta) + (1-alpha) * Lhat ...")
    R_comp = np.clip(alpha * (SE @ beta) + (1.0 - alpha) * Lhat, 0.0, 1.0)

    # Build LinUCB from full data using composite reward
    print("[*] Building LinUCB sufficient stats from synthetic data...")
    A_mats, b_vecs, counts = build_linucb_stats(X, A, R_comp, N_ACTIONS, DIM, reg=1.0)

    # Save a couple artifacts
    dump(rf, os.path.join(OUTDIR, "rf_demo.joblib"))
    with open(os.path.join(OUTDIR, "meta.json"), "w") as f:
        json.dump({"alpha": alpha, "beta": beta.tolist(), "r2_rf": r2, "n_rounds": N_ROUNDS}, f, indent=2)

    # Prepare bandit_feedback dict for OPE (as obp expects)
    # Use reward = R_comp (we backfill using our composite for offline evaluation)
    bandit_feedback = {
        "n_rounds": int(N_ROUNDS),
        "n_actions": int(N_ACTIONS),
        "context": X,
        "action": A,
        "reward": R_comp,
        "pscore": P
    }

    print("[*] Computing target policy distributions...")
    # uniform action-dist
    ad_uniform = uniform_action_dist(N_ROUNDS, N_ACTIONS)
    # linucb target action distr from stats (softmax over scores)
    ad_linucb = linucb_action_dist_from_stats(A_mats, b_vecs, LINUCb_ALPHA, X)

    # Run OPE
    print("[*] Running offline policy evaluation (IPS / SNIPS / DR)...")
    est_u, est_l, uplift = run_ope_and_report(bandit_feedback, ad_uniform, ad_linucb)

    # Plot cumulative expected reward (approx)
    out_plot = os.path.join(OUTDIR, "cum_reward_offline.png")
    plot_cumulative_expected(bandit_feedback, ad_uniform, ad_linucb, out_plot)

    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rf_r2": r2,
        "alpha": alpha,
        "beta": beta.tolist(),
        "ope_uniform": {"ips": float(est_u["ipw"]), "snip": float(est_u["snip"]), "dr": float(est_u["dr"])},
        "ope_linucb": {"ips": float(est_l["ipw"]), "snip": float(est_l["snip"]), "dr": float(est_l["dr"])},
        "uplift_dr": uplift
    }
    with open(os.path.join(OUTDIR, "summary_offline.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[*] Summary saved to", os.path.join(OUTDIR, "summary_offline.json"))
    print("[*] Done.")

if __name__ == "__main__":
    main()