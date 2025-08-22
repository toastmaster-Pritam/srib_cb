# demo_server.py
import os, hashlib, numpy as np
from datetime import datetime, timezone
from joblib import load
from flask import Flask, request, jsonify
from threading import RLock
from sqlalchemy import desc
from db import SessionLocal, ModelVersion, LinUCBSufficientStats, TrafficAllocation, EventLog
from np_ser import dumps_np, loads_np

app = Flask(__name__)
_lock = RLock()
STATE = {"alloc": [], "versions": {}}

def load_state():
    s = SessionLocal()
    try:
        STATE["alloc"].clear()
        STATE["versions"].clear()
        alloc_rows = s.query(TrafficAllocation).filter_by(active=1).order_by(TrafficAllocation.bucket_start).all()
        for a in alloc_rows:
            mv = s.query(ModelVersion).filter_by(id=a.version_id).one()
            STATE["alloc"].append({
                "version_id": mv.id, "policy_type": mv.policy_type,
                "bucket_start": a.bucket_start, "bucket_end": a.bucket_end
            })
            if mv.id not in STATE["versions"]:
                entry = {"policy_type": mv.policy_type, "n_actions": mv.n_actions, "d": mv.dim_context,
                         "alpha_lin": mv.alpha_lin, "alpha_reward": mv.alpha_reward,
                         "beta_reward": np.array(mv.beta_reward or [1,0,0], dtype=float),
                         "A": {}, "b": {}, "rf": None}
                if mv.policy_type == "linucb":
                    stats = s.query(LinUCBSufficientStats).filter_by(version_id=mv.id).all()
                    for st in stats:
                        try:
                            entry["A"][st.action_id] = loads_np(st.A_blob)
                            entry["b"][st.action_id] = loads_np(st.b_blob)
                        except Exception:
                            entry["A"][st.action_id] = np.eye(entry["d"])
                            entry["b"][st.action_id] = np.zeros(entry["d"])
                STATE["versions"][mv.id] = entry
        print(f"[server] loaded state: allocations={len(STATE['alloc'])} versions={len(STATE['versions'])}")
    finally:
        s.close()

with _lock:
    load_state()

def bucket(user_id: str) -> float:
    h = hashlib.sha256((user_id or "anon").encode()).hexdigest()
    return (int(h[:8], 16) % 10_000_000) / 10_000_000.0

def choose_version(user_id: str):
    u = bucket(user_id)
    for a in STATE["alloc"]:
        if a["bucket_start"] <= u < a["bucket_end"]:
            return a
    return STATE["alloc"][-1]

def linucb_scores(ver, x):
    d = ver["d"]; xcol = x.reshape(d,1)
    scores = np.zeros(ver["n_actions"])
    for a in range(ver["n_actions"]):
        A = ver["A"].get(a, np.eye(d))
        b = ver["b"].get(a, np.zeros(d))
        Ainv = np.linalg.inv(A)
        theta = (Ainv @ b.reshape(d,1))
        mean = float((theta.T @ xcol).item())
        std = float(np.sqrt((xcol.T @ Ainv @ xcol).item()))
        scores[a] = mean + ver["alpha_lin"] * std
    return scores

@app.route("/health")
def health():
    return jsonify({"ok": True, "alloc": STATE["alloc"]})

@app.route("/allocations")
def allocations():
    return jsonify({"alloc": STATE["alloc"]})

@app.route("/reload", methods=["POST"])
def reload():
    with _lock:
        load_state()
    return jsonify({"ok": True})

@app.route("/recommend", methods=["POST"])
def recommend():
    payload = request.get_json(force=True)
    ctx = np.array(payload["context"], dtype=float).reshape(-1)
    user_id = payload.get("user_id", f"anon-{np.random.randint(1e9)}")
    chosen = choose_version(user_id)
    version_id = chosen["version_id"]
    ver = STATE["versions"][version_id]

    if ver["policy_type"] == "uniform":
        probs = np.ones(ver["n_actions"]) / ver["n_actions"]
        action = int(np.random.choice(ver["n_actions"], p=probs))
        scores = np.zeros(ver["n_actions"])
    else:
        scores = linucb_scores(ver, ctx)
        ex = np.exp(scores - np.max(scores)); probs = ex / ex.sum()
        action = int(np.random.choice(ver["n_actions"], p=probs))

    s = SessionLocal()
    try:
        ev = EventLog(user_id=user_id, request_id=payload.get("request_id"), version_id=version_id,
                      context_blob=dumps_np(ctx), action_id=action, pscore=float(probs[action]))
        s.add(ev); s.commit()
        event_id = ev.id
    finally:
        s.close()
    print(f"[recommend] uid={user_id[:12]} ver={ver['policy_type']} vid={version_id} action={action} p={probs[action]:.3f} ev={event_id}")
    return jsonify({"version_id": version_id, "policy": ver["policy_type"], "action": action,
                    "scores": scores.tolist(), "probs": probs.tolist(), "event_id": event_id})

@app.route("/log", methods=["POST"])
def log():
    payload = request.get_json(force=True)
    ev_id = int(payload["event_id"])
    short = payload.get("short_events", [0,0,0])
    reward = payload.get("reward", None)
    long_term = payload.get("long_term", None)
    s = SessionLocal()
    try:
        ev = s.query(EventLog).filter_by(id=ev_id).one()
        ev.click = int(short[0]); ev.revisit = int(short[1]); ev.watch_time = float(short[2])
        if reward is not None:
            ev.reward = float(reward)
        if long_term is not None:
            ev.long_term = float(long_term)
        s.commit()
    except Exception as e:
        s.rollback(); return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        s.close()
    print(f"[log] ev={ev_id} click={short[0]} revisit={short[1]} watch={short[2]:.3f} reward={reward} lt={long_term}")
    return jsonify({"ok": True})

if __name__=="__main__":
    app.run(host=os.getenv("APP_HOST","0.0.0.0"), port=int(os.getenv("APP_PORT",8000)))
