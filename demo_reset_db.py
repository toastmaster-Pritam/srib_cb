# demo_reset_db.py
import os, shutil
from db import SessionLocal, ModelVersion, LinUCBSufficientStats, TrafficAllocation, EventLog, DailyMetrics, init_db
from sqlalchemy import delete

def reset_db(confirm=False):
    if not confirm:
        print("Pass confirm=True to actually wipe. Example: python3 demo_reset_db.py confirm")
        return
    s = SessionLocal()
    try:
        s.execute(delete(LinUCBSufficientStats))
        s.execute(delete(TrafficAllocation))
        s.execute(delete(EventLog))
        s.execute(delete(DailyMetrics))
        s.execute(delete(ModelVersion))
        s.commit()
        print("All demo tables truncated.")
    finally:
        s.close()

def clear_model_store():
    if os.path.exists("model_store"):
        shutil.rmtree("model_store")
    os.makedirs("model_store", exist_ok=True)
    print("model_store cleared.")

if __name__ == "__main__":
    import sys
    init_db()
    confirm_flag = ("confirm" in sys.argv)
    reset_db(confirm=confirm_flag)
    clear_model_store()
    print("Demo DB reset complete.")
