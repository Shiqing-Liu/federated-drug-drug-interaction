#!/usr/bin/env bash
# Run the 4 experiments sequentially, then plot + summarize.
# Runs whose CSV is already complete are skipped (main.py checks), so you can
# just re-run this after a Colab disconnect to resume.
set -e

ROUNDS="${ROUNDS:-50}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-4}"
K="${K:-4}"
SEED="${SEED:-0}"
OUT="${OUT:-results/}"

COMMON="--rounds $ROUNDS --local_epochs $LOCAL_EPOCHS --num_clients $K --seed $SEED --out $OUT"

echo "=== 1/4 Centralized GraphSAGE (upper bound) ==="
python main.py --mode centralized --encoder sage $COMMON

echo "=== 2/4 Federated GraphSAGE (main result) ==="
python main.py --mode federated --encoder sage $COMMON

echo "=== 3/4 Federated GCN (vanilla GCN baseline) ==="
python main.py --mode federated --encoder gcn $COMMON

echo "=== 4/4 Local-only GraphSAGE (no collaboration) ==="
python main.py --mode local --encoder sage $COMMON

echo "=== Plot + summary ==="
python plot.py --out "$OUT" --num_clients "$K" --seed "$SEED"
python summarize.py --out "$OUT" --num_clients "$K" --seed "$SEED"
echo "Done. See $OUT/hits20.png and $OUT/summary.md"
