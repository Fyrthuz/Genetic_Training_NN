#!/usr/bin/env bash
# Limpia resultados, reejecuta el experimento completo y levanta TensorBoard.
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
LOGDIR="$ROOT/results/tb"
RUN_LOG="$ROOT/results/run_log.txt"
TB_LOG="$ROOT/results/tensorboard.log"
PORT="${TB_PORT:-6006}"

echo "==> Parando procesos previos (experimento + tensorboard)..."
pkill -f "src.run_experiments" 2>/dev/null || true
pkill -f "tensorboard.*results/tb" 2>/dev/null || true
sleep 2

echo "==> Limpiando resultados previos..."
rm -rf "$ROOT/results"
mkdir -p "$LOGDIR" "$ROOT/plots"

echo "==> Arrancando el experimento (8 configs x 3 semillas) en background..."
setsid nohup uv run python -u -m src.run_experiments > "$RUN_LOG" 2>&1 < /dev/null &
EXP_PID=$!
disown
echo "    Experimento PID=$EXP_PID  (log: $RUN_LOG)"

echo "==> Arrancando TensorBoard en http://localhost:$PORT ..."
setsid nohup uv run tensorboard --logdir="$LOGDIR" --port "$PORT" \
    --host 0.0.0.0 > "$TB_LOG" 2>&1 < /dev/null &
TB_PID=$!
disown
echo "    TensorBoard PID=$TB_PID  (log: $TB_LOG)"

echo ""
echo "Todo listo:"
echo "  - TensorBoard: http://localhost:$PORT"
echo "  - Log experimento: $RUN_LOG"
echo "  - Para parar: pkill -f 'src.run_experiments' && pkill -f tensorboard"
