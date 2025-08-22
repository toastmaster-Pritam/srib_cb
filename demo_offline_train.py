# demo_offline_train.py
import os, numpy as np, torch
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from db import SessionLocal, ModelVersion, LinUCBSufficientStats, TrafficAllocation
from np_ser import dumps_np
from obp.dataset import SyntheticBanditDataset

SEED = 42
N_ACTIONS = 10
DIM = 12
N_ROUNDS = 3000   # smaller for demo
REG_LAMBDA = 1.0
ALPHA_LIN = 0.15

def make_uniform(session):
    existing = session.query(ModelVersion).filter_by(policy_type="uniform").first()
    if existing:
        return existing.id
    mv = ModelVersion(policy_type="uniform", n_actions=N_ACTIONS, dim_context=DIM,
                      alpha_lin=ALPHA_LIN, alpha_reward=0.0, beta_reward=[0,0,0], notes="control uniform")
    session.add(mv); session.flush()
    I = np.eye(DIM)*REG_LAMBDA; z = np.zeros(DIM)
    for a in range(N_ACTIONS):
        st = LinUCBSufficientStats(version_id=mv.id, action_id=a, A_blob=dumps_np(I), b_blob=dumps_np(z), count=0)
        session.add(st)
    session.commit()
    return mv.id

def main():
    print("[bootstrap] Generating synthetic bandit logs (demo)...")
    ds = SyntheticBanditDataset(n_actions=N_ACTIONS, dim_context=DIM, reward_type="binary", random_state=SEED)
    bf = ds.obtain_batch_bandit_feedback(n_rounds=N_ROUNDS)
    X = bf["context"]; A = bf["action"]

    rng = np.random.default_rng(SEED)
    clicks = rng.binomial(1,0.30,size=N_ROUNDS)
    revis = rng.binomial(1,0.20,size=N_ROUNDS)
    wtime = rng.random(N_ROUNDS)
    SE = np.stack([clicks,revis,wtime],axis=1)

    ctx_sig = (X[:,0]-X[:,1]); ctx_sig = ctx_sig/max(1.0, np.std(ctx_sig))
    LT = 0.4*wtime + 0.25*clicks + 0.15*revis + 0.2*ctx_sig
    LT = np.clip(LT + rng.normal(0,0.03,size=N_ROUNDS), 0, 1)

    Xrf = np.hstack([X, SE])
    rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    rf.fit(Xrf, LT)
    LT_hat = rf.predict(Xrf)

    # learn alpha/beta quickly
    se_t = torch.tensor(SE, dtype=torch.float32)
    lt_hat_t = torch.tensor(LT_hat, dtype=torch.float32)
    lt_t = torch.tensor(LT, dtype=torch.float32)
    a_raw = torch.tensor(0.0, requires_grad=True)
    b_raw = torch.tensor([0.34,0.22,0.44], requires_grad=True)
    opt = torch.optim.Adam([a_raw,b_raw], lr=0.02)
    for _ in range(200):
        opt.zero_grad()
        alpha = torch.sigmoid(a_raw)
        beta = torch.nn.functional.softmax(b_raw, dim=0)
        r_short = se_t @ beta
        r_comp = alpha * r_short + (1.0 - alpha) * lt_hat_t
        loss = torch.mean((r_comp - lt_t)**2) + 1e-4*(alpha - 0.5)**2
        loss.backward(); opt.step()
    alpha_val = float(torch.sigmoid(a_raw).item())
    beta_val = torch.softmax(b_raw.detach(),dim=0).numpy()
    print("[bootstrap] learned alpha", alpha_val, "beta", beta_val)

    R = np.clip(alpha_val * (SE @ beta_val) + (1-alpha_val) * LT_hat, 0.0, 1.0)

    A_mats = np.stack([np.eye(DIM)*REG_LAMBDA for _ in range(N_ACTIONS)])
    b_vecs = np.zeros((N_ACTIONS, DIM)); counts = np.zeros(N_ACTIONS, dtype=int)
    for x,a,r in zip(X, A, R):
        A_mats[a] += np.outer(x,x)
        b_vecs[a] += r * x
        counts[a] += 1

    os.makedirs("model_store", exist_ok=True)
    rf_path = f"model_store/rf_v{np.random.randint(1e9)}.joblib"; dump(rf, rf_path)

    session = SessionLocal()
    try:
        ctrl_id = make_uniform(session)
        mv = ModelVersion(policy_type="linucb", n_actions=N_ACTIONS, dim_context=DIM,
                          alpha_lin=ALPHA_LIN, alpha_reward=alpha_val, beta_reward=beta_val.tolist(),
                          notes=f"bootstrap rf_path={rf_path}")
        session.add(mv); session.flush()
        for a in range(N_ACTIONS):
            st = LinUCBSufficientStats(version_id=mv.id, action_id=a, A_blob=dumps_np(A_mats[a]),
                                      b_blob=dumps_np(b_vecs[a]), count=int(counts[a]))
            session.add(st)

        # DEMO: 50% control, 50% candidate
        session.add(TrafficAllocation(version_id=ctrl_id, bucket_start=0.0, bucket_end=0.5, active=1))
        session.add(TrafficAllocation(version_id=mv.id, bucket_start=0.5, bucket_end=1.0, active=1))
        session.commit()
        print("[bootstrap] Created linucb version:", mv.id, "with 50/50 allocation")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

if __name__=="__main__":
    main()
