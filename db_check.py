# db_check.py
import os
from db import SessionLocal, ModelVersion, TrafficAllocation, EventLog, init_db

print("DATABASE_URL (from env):", os.environ.get("DATABASE_URL"))
init_db()  # safe: creates tables if not existing
s = SessionLocal()
try:
    mv_count = s.query(ModelVersion).count()
    alloc_count = s.query(TrafficAllocation).count()
    ev_count = s.query(EventLog).count()
    print("ModelVersion rows:", mv_count)
    print("TrafficAllocation rows:", alloc_count)
    print("EventLog rows:", ev_count)
    print("\nActive allocations (active=1):")
    for a in s.query(TrafficAllocation).filter_by(active=1).order_by(TrafficAllocation.bucket_start).all():
        mv = s.query(ModelVersion).filter_by(id=a.version_id).one_or_none()
        print("  version_id:", a.version_id, "policy:", getattr(mv,"policy_type",None),
              "range:", a.bucket_start, "-", a.bucket_end)
    print("\nLast 10 events (id, ts, version_id, action, pscore, click, reward, long_term):")
    rows = s.query(EventLog).order_by(EventLog.ts.desc()).limit(10).all()
    for r in rows:
        print(r.id, r.ts, r.version_id, r.action_id, r.pscore, r.click, r.reward, r.long_term)
finally:
    s.close()
