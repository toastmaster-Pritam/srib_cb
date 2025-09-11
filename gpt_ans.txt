"""
api.py - FastAPI endpoints for the ModelService with feature-pool and context-instance support.

Run:
    uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import uvicorn

from model_service import ModelService

app = FastAPI(title="ContextualBanditModelAPI")
svc = ModelService()

# -------------------------
# Request models
# -------------------------
class SelectFeaturesReq(BaseModel):
    features: List[str]

class InstanceReq(BaseModel):
    instance: Dict[str, Any]

class MultipleInstancesReq(BaseModel):
    instances: List[Dict[str, Any]]

class PredictInstanceReq(BaseModel):
    instance_id: Optional[int] = None
    instance: Optional[Dict[str, Any]] = None
    n_actions: Optional[int] = 1
    stochastic: Optional[bool] = False

class SimulateReq(BaseModel):
    instance_ids: Optional[List[int]] = None
    horizon: Optional[int] = 60
    n_actions: Optional[int] = 3
    n_trials: Optional[int] = 3
    context_mode: Optional[str] = "random"
    propensity_mc: Optional[int] = 500

class UpdateEntry(BaseModel):
    a: Any
    trace: List[int]
    context: Optional[Dict[str, Any]] = None
    instance_id: Optional[int] = None
    t: Optional[int] = None
    p_b: Optional[float] = 0.0

class UpdateBatchReq(BaseModel):
    updates: List[UpdateEntry]

# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/features")
def get_features():
    try:
        return {"features": svc.get_feature_pool(), "selected": svc.get_selected_features()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/select_features")
def set_selected_features(req: SelectFeaturesReq):
    try:
        svc.set_selected_features(req.features)
        return {"selected": svc.get_selected_features()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/instances")
def get_instances():
    try:
        return {"instances": svc.get_context_instances()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instances")
def add_instance(req: InstanceReq):
    try:
        idx = svc.add_context_instance(req.instance)
        return {"instance_id": idx}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate_instances")
def generate_instances(n: int = 5):
    try:
        ids = svc.generate_random_instances(n=n)
        return {"added_ids": ids}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_instance")
def predict_instance(req: PredictInstanceReq):
    try:
        if req.instance_id is not None:
            actions = svc.predict(instance_id=req.instance_id, n_actions=req.n_actions, stochastic=req.stochastic)
        elif req.instance is not None:
            actions = svc.predict(instance=req.instance, n_actions=req.n_actions, stochastic=req.stochastic)
        else:
            actions = svc.predict(n_actions=req.n_actions, stochastic=req.stochastic)
        return {"actions": actions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/simulate")
def simulate(req: SimulateReq):
    try:
        res = svc.simulate_trials(n_trials=req.n_trials, horizon=req.horizon, n_actions=req.n_actions, propensity_mc=req.propensity_mc, selected_instance_ids=req.instance_ids, context_mode=req.context_mode)
        return {
            "cumavg_mean": res["cumavg_mean"].tolist(),
            "cumavg_std": res["cumavg_std"].tolist(),
            "regrets_shape": list(res["regrets"].shape),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/update_batch")
def update_batch(req: UpdateBatchReq):
    try:
        updates = [u.dict() for u in req.updates]
        n = svc.batch_update(updates, log=True)
        return {"updated": n}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/ope")
def ope(n_boot: int = 200):
    try:
        metrics = svc.compute_ope(n_boot=n_boot)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# dev-run
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)