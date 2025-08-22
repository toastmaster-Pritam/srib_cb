# demo_monitor.py
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from db import SessionLocal, EventLog, ModelVersion, TrafficAllocation

OUT = "monitor_outputs"; os.makedirs(OUT, exist_ok=True)

def load_joined(session, since_hours=48):
    since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    q = session.query(EventLog.id, EventLog.ts, EventLog.version_id, EventLog.action_id, EventLog.click, EventLog.reward, EventLog.long_term, ModelVersion.policy_type).join(
        ModelVersion, ModelVersion.id == EventLog.version_id
    ).filter(EventLog.ts >= since)
    rows = q.all()
    if not rows:
        return pd.DataFrame(columns=["id","ts","version_id","action_id","click","reward","long_term","policy_type"])
    df = pd.DataFrame([{
        "id": r.id, "ts": r.ts, "version_id": r.version_id, "action_id": r.action_id,
        "click": r.click, "reward": r.reward, "long_term": r.long_term, "policy_type": r.policy_type
    } for r in rows])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    return df

def summary(df):
    if df.empty:
        print("No events to summarize.")
        return
    agg = df.groupby("policy_type").agg(impressions=("id","count"), clicks=("click","sum"), avg_reward=("reward","mean")).reset_index()
    agg["ctr"] = agg["clicks"]/agg["impressions"]
    print("\nSummary by policy:\n", agg)
    return agg

def plot_cumulative(df):
    if df.empty: return
    df = df.sort_values("ts")
    df["reward_filled"] = df["reward"].fillna(0.0)
    plt.figure(figsize=(8,4))
    for name, g in df.groupby("policy_type"):
        g = g.sort_values("ts")
        g["cum_reward"] = g["reward_filled"].cumsum()
        plt.plot(g["ts"], g["cum_reward"], label=name)
    plt.legend(); plt.title("Cumulative reward by policy"); plt.tight_layout()
    out = os.path.join(OUT, "cum_reward.png"); plt.savefig(out); plt.close()
    print("Saved", out)

def plot_daily_ctr(df):
    if df.empty: return
    df["date"] = df["ts"].dt.date
    daily = df.groupby(["date","policy_type"]).agg(impressions=("id","count"), clicks=("click","sum")).reset_index()
    daily["ctr"] = daily["clicks"]/daily["impressions"]
    pivot = daily.pivot(index="date", columns="policy_type", values="ctr").fillna(0)
    plt.figure(figsize=(8,4))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker='o', label=col)
    plt.legend(); plt.title("Daily CTR by policy"); plt.tight_layout()
    out = os.path.join(OUT, "daily_ctr.png"); plt.savefig(out); plt.close()
    print("Saved", out)

def show_allocations(session):
    rows = session.query(TrafficAllocation).filter_by(active=1).order_by(TrafficAllocation.bucket_start).all()
    print("\nActive allocations:")
    for r in rows:
        mv = session.query(ModelVersion).filter_by(id=r.version_id).one()
        print(f" version {r.version_id} ({mv.policy_type}) : {r.bucket_start:.2f} - {r.bucket_end:.2f}")

def main():
    s = SessionLocal()
    try:
        df = load_joined(s, since_hours=48)
        print("Total events loaded:", len(df))
        summary(df)
        plot_cumulative(df)
        plot_daily_ctr(df)
        show_allocations(s)
    finally:
        s.close()

if __name__=="__main__":
    main()
