#!/usr/bin/env bash
set -euo pipefail
LOGDIR="logs"
mkdir -p "$LOGDIR"
mkdir -p "model_store"
mkdir -p "monitor_outputs"

APP_HOST="${APP_HOST:-localhost}"
APP_PORT="${APP_PORT:-8000}"

echo "===== DEMO RUN START ====="
date -u

echo; echo "[1/6] Reset DB and model_store..."
python3 demo_reset_db.py confirm | tee "$LOGDIR/reset.log"

echo; echo "[2/6] Offline bootstrap (create uniform + linucb)..."
python3 demo_offline_train.py | tee "$LOGDIR/bootstrap.log"

echo; echo "[3/6] Start demo_server.py..."
nohup python3 demo_server.py > "$LOGDIR/server.log" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID (logs:$LOGDIR/server.log)"
# wait small loop for /health
for i in $(seq 1 15); do
  if curl -s "http://$APP_HOST:$APP_PORT/health" | grep -q '"ok"'; then
    echo "server healthy."
    break
  fi
  sleep 1
  if [ $i -eq 15 ]; then
    echo "server didn't respond; tail server log:"
    tail -n 200 "$LOGDIR/server.log"
    kill $SERVER_PID || true
    exit 1
  fi
done

echo; echo "[4/6] Running simulator (background)..."
nohup python3 demo_simulator.py --n_requests 500 --sleep 0.001 --lt_prob 0.30 --reward_prob 0.45 > "$LOGDIR/sim.log" 2>&1 &
SIM_PID=$!
echo "sim pid=$SIM_PID (logs:$LOGDIR/sim.log)"

# wait for simulator to finish by polling process
echo "Waiting for simulator to finish..."
wait $SIM_PID || true
echo "Simulator finished."

echo; echo "[5/6] Run demo_worker.py (retrain & update allocations)"
python3 demo_worker.py | tee "$LOGDIR/worker.log"

echo; echo "[6/6] Generate monitoring plots..."
python3 demo_monitor.py | tee "$LOGDIR/monitor.log"

echo; echo "Active allocations (server /allocations):"
curl -s "http://$APP_HOST:$APP_PORT/allocations" || true
echo

echo "=== tail server.log ==="
tail -n 200 "$LOGDIR/server.log" || true
echo "=== tail sim.log ==="
tail -n 200 "$LOGDIR/sim.log" || true
echo "=== tail worker.log ==="
tail -n 200 "$LOGDIR/worker.log" || true

echo "Shutting down server (pid $SERVER_PID)..."
kill "$SERVER_PID" 2>/dev/null || true
sleep 1

echo "Demo finished. Check monitor_outputs/ and logs/"
date -u
