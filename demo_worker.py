# demo_worker.py
import os, time, numpy as np, torch, requests
from datetime import datetime, timezone
from joblib import dump
from sqlalchemy import desc
from db import SessionLocal, ModelVersion, LinUCBSufficientStats, TrafficAllocation, EventLog
from np_ser import dumps_np, loads_np
from sklearn.ensemble import RandomForestRegressor

# demo-friendly settings
REG = 1.0
ALPHA_LIN_DEFAULT = 0.15
PROMOTE_STEP = 0.20
DEMOTE_STEP = 0.20
UPLIFT_THRESHOLD = 0.01
MIN_LT_LABELS = 10
MIN_CONTROL_LOGS = 30
MIN_TOTAL_EVENTS = 50
DEMO_ITERATIONS = 3
SLEEP_BETWEEN = 2

APP_HOST = os.getenv("APP_HOST","localhost")
APP_PORT = int(os.getenv("APP_PORT",8000))
RELOAD_URL = f"http://{APP_HOST}:{APP_PORT}/reload"

def rows_to_arrays(rows):
    X=[]; A=[]; R=[]; P=[]; V=[]; SE=[]; LT=[]
    for r in rows:
        X.append(loads_np(r.context_blob))
        A.append(r.action_id)
        R.append(r.reward if r.reward is not None else np.nan)
        P.append(r.pscore if r.pscore is not None else np.nan)
        V.append(r.version_id)
        SE.append([r.click or 0, r.revisit or 0, r.watch_time or 0.0])
        LT.append(r.long_term if r.long_term is not None else np.nan)
    return np.array(X), np.array(A), np.array(R), np.array(P), np.array(V), np.array(SE), np.array(LT)

def build_linucb(X,A,R,n_actions,d):
    A_mats = np.stack([np.eye(d)*REG for _ in range(n_actions)])
    b_vecs = np.zeros((n_actions,d)); counts=np.zeros(n_actions,dtype=int)
    for x,a,r in zip(X,A,R):
        A_mats[a]+=np.outer(x,x); b_vecs[a]+=r*x; counts[a]+=1
    return A_mats,b_vecs,counts

def trigger_reload():
    try:
        r = requests.post(RELOAD_URL, timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False

def main():
    s = SessionLocal()
    try:
        rows = s.query(EventLog).order_by(desc(EventLog.ts)).limit(50000).all()
        if not rows:
            print("[worker] no rows - run simulator first")
            return
        X,A,R,P,V,SE,LT = rows_to_arrays(rows)
        print(f"[worker] loaded {len(X)} events, {int((~np.isnan(LT)).sum())} long-term labels")

        if len(X) < MIN_TOTAL_EVENTS:
            print("[worker] not enough events:", len(X)); return

        for it in range(DEMO_ITERATIONS):
            print(f"[worker] iter {it+1}/{DEMO_ITERATIONS} at {datetime.now(timezone.utc).isoformat()}")
            rows = s.query(EventLog).order_by(desc(EventLog.ts)).limit(50000).all()
            X,A,R,P,V,SE,LT = rows_to_arrays(rows)
            mask_lt = ~np.isnan(LT)
            if mask_lt.sum() < MIN_LT_LABELS:
                print(f"[worker] not enough LT labels ({mask_lt.sum()}) - waiting {SLEEP_BETWEEN}s"); time.sleep(SLEEP_BETWEEN); continue

            Xrf = np.hstack([X[mask_lt], SE[mask_lt]])
            y = LT[mask_lt]
            rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=42)
            rf.fit(Xrf, y)
            lt_hat_all = rf.predict(np.hstack([X, SE]))

            # alpha/beta quick fit
            se_t = torch.tensor(SE[mask_lt], dtype=torch.float32)
            lt_hat_t = torch.tensor(rf.predict(np.hstack([X[mask_lt], SE[mask_lt]])), dtype=torch.float32)
            lt_true_t = torch.tensor(y, dtype=torch.float32)
            a_raw = torch.tensor(0.0, requires_grad=True)
            b_raw = torch.tensor([0.4,0.2,0.4], requires_grad=True)
            opt = torch.optim.Adam([a_raw,b_raw], lr=0.02)
            for _ in range(200):
                opt.zero_grad()
                alpha = torch.sigmoid(a_raw)
                beta = torch.nn.functional.softmax(b_raw, dim=0)
                r_short = se_t @ beta
                r_comp = alpha * r_short + (1 - alpha) * lt_hat_t
                loss = torch.mean((r_comp - lt_true_t)**2) + 1e-4*(alpha-0.5).pow(2)
                loss.backward(); opt.step()
            alpha_val = float(torch.sigmoid(a_raw).item())
            beta_val = torch.softmax(b_raw.detach(), dim=0).numpy()
            print("[worker] alpha,beta:", alpha_val, beta_val)

            R_comp = np.clip(alpha_val * (SE @ beta_val) + (1 - alpha_val) * lt_hat_all, 0.0, 1.0)

            # backfill rewards
            updated = 0
            for row_obj, r_val in zip(rows, R_comp):
                if row_obj.reward is None:
                    row_obj.reward = float(r_val); updated += 1
            s.commit()
            print(f"[worker] backfilled {updated} reward fields")

            # rebuild LinUCB
            n_actions = int(s.query(ModelVersion).filter_by(policy_type="linucb").first().n_actions)
            d = int(s.query(ModelVersion).filter_by(policy_type="linucb").first().dim_context)
            A_mats,b_vecs,counts = build_linucb(X,A,R_comp,n_actions,d)

            # save new candidate version
            os.makedirs("model_store", exist_ok=True)
            rf_path = f"model_store/rf_v{np.random.randint(1e9)}.joblib"; dump(rf, rf_path)
            new_mv = ModelVersion(policy_type="linucb", n_actions=n_actions, dim_context=d,
                                  alpha_lin=ALPHA_LIN_DEFAULT, alpha_reward=alpha_val, beta_reward=beta_val.tolist(),
                                  notes=f"demo worker rf_path={rf_path}")
            s.add(new_mv); s.flush()
            for a in range(n_actions):
                st = LinUCBSufficientStats(version_id=new_mv.id, action_id=a,
                                           A_blob=dumps_np(A_mats[a]), b_blob=dumps_np(b_vecs[a]), count=int(counts[a]))
                s.add(st)
            s.commit()
            print("[worker] rolled new version", new_mv.id)

            # Evaluate vs control logs (if we have enough)
            ctrl = s.query(ModelVersion).filter_by(policy_type="uniform").order_by(ModelVersion.id).first()
            mask_ctrl = (V == ctrl.id)
            if mask_ctrl.sum() < MIN_CONTROL_LOGS:
                print("[worker] not enough control logs for OPE:", int(mask_ctrl.sum()))
            else:
                # simple uplift sign: use mean reward on control vs candidate predicted DR using naive replacement
                ctrl_mean = float(np.nanmean(R_comp[mask_ctrl]))
                cand_mean = float(np.nanmean(R_comp))  # quick proxy
                uplift = cand_mean - ctrl_mean
                print(f"[worker] quick ctrl_mean={ctrl_mean:.4f} cand_mean={cand_mean:.4f} uplift={uplift:.4f}")

                # update allocation: simple promote/demote control vs candidate
                allocs = s.query(TrafficAllocation).filter_by(active=1).all()
                p_candidate = 0.5
                current = [a for a in allocs if a.version_id == new_mv.id]
                if current:
                    p_candidate = current[0].bucket_end - current[0].bucket_start
                if uplift > UPLIFT_THRESHOLD and p_candidate < 0.95:
                    p_candidate = min(0.95, p_candidate + PROMOTE_STEP)
                    print("[worker] promote", p_candidate)
                elif uplift < -UPLIFT_THRESHOLD and p_candidate > 0.01:
                    p_candidate = max(0.01, p_candidate - DEMOTE_STEP)
                    print("[worker] demote", p_candidate)
                else:
                    print("[worker] no allocation change")

                for a in allocs:
                    a.active = 0
                s.add(TrafficAllocation(version_id=ctrl.id, bucket_start=0.0, bucket_end=1.0-p_candidate, active=1))
                s.add(TrafficAllocation(version_id=new_mv.id, bucket_start=1.0-p_candidate, bucket_end=1.0, active=1))
                s.commit()
                print("[worker] updated allocations in DB")
                ok = trigger_reload()
                print("[worker] triggered server reload:", ok)
                allocs_now = s.query(TrafficAllocation).filter_by(active=1).order_by(TrafficAllocation.bucket_start).all()
                for a in allocs_now:
                    mv = s.query(ModelVersion).filter_by(id=a.version_id).one()
                    print(f"[worker] active version {a.version_id} ({mv.policy_type}): {a.bucket_start:.2f}-{a.bucket_end:.2f}")

            time.sleep(SLEEP_BETWEEN)
    finally:
        s.close()

if __name__=="__main__":
    main()
