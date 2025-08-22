#!/usr/bin/env python3
# demo_simulator.py
import time, uuid, requests, numpy as np, argparse
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

rng = np.random.default_rng(123)
API_HOST = "localhost"
API_PORT = 8000

def gen_context(D=12):
    return rng.normal(0, 1, size=D).tolist()

def gen_short(action):
    base_click = 0.08 + 0.04 * (action % 4)
    base_revisit = 0.04 + 0.03 * ((action + 1) % 4)
    click = int(rng.random() < base_click)
    revisit = int(rng.random() < base_revisit)
    watch = float(np.clip(rng.normal(0.45 + 0.02 * action, 0.12), 0, 1))
    return [click, revisit, watch]

def make_session():
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.2, status_forcelist=(500,502,503,504))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def run(n_requests=1000, sleep=0.001, lt_prob=0.25, reward_prob=0.4, timeout=3.0, print_every=100):
    api = f"http://{API_HOST}:{API_PORT}"
    s = make_session()
    # wait for health
    try:
        ok = s.get(f"{api}/health", timeout=2.0).json().get("ok", False)
    except Exception:
        ok = False
    if not ok:
        print("[sim] server /health not ready; continuing anyway")
    sent=0; rec_err=0; log_err=0; succ=0
    for i in tqdm(range(n_requests), desc="sending"):
        user = f"user-{rng.integers(1_000_000)}"
        req_id = str(uuid.uuid4()); ctx = gen_context()
        try:
            r = s.post(f"{api}/recommend", json={"user_id":user,"context":ctx,"request_id":req_id}, timeout=timeout)
            r.raise_for_status()
            recj = r.json()
            ev_id = recj["event_id"]; a = recj["action"]
        except Exception as e:
            rec_err += 1
            if (i+1) % print_every == 0:
                print(f"[sim] {i+1}/{n_requests} recommend_err={rec_err}")
            time.sleep(sleep)
            continue

        short = gen_short(a)
        payload = {"event_id": ev_id, "short_events": short}
        if rng.random() < lt_prob:
            payload["long_term"] = float(np.clip(0.2*short[0] + 0.2*short[1] + 0.6*short[2] + rng.normal(0,0.03),0,1))
        if rng.random() < reward_prob:
            payload["reward"] = float(0.3*short[0] + 0.2*short[1] + 0.5*short[2])

        try:
            r2 = s.post(f"{api}/log", json=payload, timeout=timeout)
            r2.raise_for_status()
            succ += 1
        except Exception as e:
            log_err += 1
            if (i+1) % print_every == 0:
                print(f"[sim] {i+1}/{n_requests} log_err={log_err}")
        sent += 1
        if sleep:
            time.sleep(sleep)
    print("=== SIM SUMMARY ===")
    print("target:", n_requests, "sent:", sent, "rec_err:", rec_err, "log_err:", log_err, "succ:", succ)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_requests", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.0005)
    parser.add_argument("--lt_prob", type=float, default=0.25)
    parser.add_argument("--reward_prob", type=float, default=0.4)
    args = parser.parse_args()
    run(n_requests=args.n_requests, sleep=args.sleep, lt_prob=args.lt_prob, reward_prob=args.reward_prob)
