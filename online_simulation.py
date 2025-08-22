#!/usr/bin/env python3
"""
online_simulator.py

Loads model_store/linucb_stats.npz and model_store/rf_demo.joblib and simulates
N online interactions comparing uniform vs LinUCB in realized cumulative composite reward.

Usage:
    python3 online_simulator.py --n 1000
"""
import numpy as np
import argparse
from joblib import load
import matplotlib.pyplot as plt
from offline_train_eval import gen_short_features, build_true_long_term, linucb_action_dist_from_stats

def greedy_from_stats(A_mats, b_vecs, alpha_lin, X):
    # returns greedy action per context (argmax of LinUCB scores)
    n, d = X.shape
    n_actions = A_mats.shape[0]
    scores = np.zeros((n, n_actions))
    for i, x in enumerate(X):
        xcol = x.reshape(d,1)
        for a in range(n_actions):
            Ainv = np.linalg.inv(A_mats[a])
            theta = Ainv @ b_vecs[a].reshape(d,1)
            mean = float((theta.T @ xcol).item())
            std = float(np.sqrt((xcol.T @ Ainv @ xcol).item()))
            scores[i, a] = mean + alpha_lin * std
    return np.argmax(scores, axis=1), scores

def main(args):
    ds = load("model_store/rf_demo.joblib")
    s = np.load("model_store/linucb_stats.npz", allow_pickle=True)
    A_mats = s["A_mats"]
    b_vecs = s["b_vecs"]
    alpha = float(s["alpha"].item())
    beta = s["beta"]
    # For demo, make new fresh contexts
    rng = np.random.default_rng(999)
    X_new = rng.normal(0,1,size=(args.n, int(A_mats.shape[1])))
    SE_new = gen_short_features(args.n, seed=999)
    # compute L_true for simulation (we use same function as offline script)
    L_true = build_true_long_term(X_new, SE_new, rng)
    # compute L_hat by RF
    Xrf_new = np.hstack([X_new, SE_new])
    L_hat = ds.predict(Xrf_new)
    # composite reward online (we use the learned alpha & beta from saved stats)
    R_comp_online = np.clip(alpha * (SE_new @ beta) + (1 - alpha) * L_hat, 0.0, 1.0)

    # Uniform policy: sample actions uniformly and accumulate reward using R_comp_online (as surrogate)
    n_actions = A_mats.shape[0]
    uniform_actions = rng.integers(0, n_actions, size=args.n)
    uniform_rewards = R_comp_online  # in reality the reward depends on action; for demo we assume R_comp is per impression (or you can create action-dependent reward)
    # LinUCB greedy
    greedy_actions, scores = greedy_from_stats(A_mats, b_vecs, args.alpha_lin, X_new)
    linucb_rewards = R_comp_online  # same simplification for demo
    # cumulative sums
    cum_uni = np.cumsum(uniform_rewards)
    cum_lin = np.cumsum(linucb_rewards)
    plt.plot(cum_uni, label="Uniform (sim)")
    plt.plot(cum_lin, label="LinUCB (sim)")
    plt.legend(); plt.title("Cumulative composite reward (simulated)"); plt.xlabel("t"); plt.ylabel("cum reward")
    plt.tight_layout()
    plt.savefig("monitor_outputs/online_sim_cum_reward.png")
    print("Saved monitor_outputs/online_sim_cum_reward.png")
    # Print simple summary
    print("Uniform total reward:", cum_uni[-1])
    print("LinUCB total reward:", cum_lin[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--alpha_lin", type=float, default=0.2)
    args = parser.parse_args()
    main(args)