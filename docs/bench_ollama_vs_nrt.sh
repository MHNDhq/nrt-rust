#!/usr/bin/env bash
# Runtime-only benchmark: NRT (Candle CPU) vs Ollama (llama.cpp), both serving
# the IDENTICAL TinyLlama 1.1B Chat v1.0 Q4_K_M GGUF file from HuggingFace.
#
# Both runtimes are warmed once before the timed loop so the reported numbers
# measure inference, not first-load cost.
#
# Usage:
#   bash docs/bench_ollama_vs_nrt.sh 2>&1 | tee docs/bench_results.txt
#
# Prerequisites:
#   1. Ollama installed and `ollama serve` running on 127.0.0.1:11434.
#   2. Ollama model `nrt-tinyllama-q4km` created from the HF GGUF.
#   3. NRT release server binary built: `cargo build --release -p nrt-server`.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
NRT_ADDR="127.0.0.1:9105"
OLLAMA_ADDR="127.0.0.1:11434"
OLLAMA_MODEL="nrt-tinyllama-q4km"
MAX_TOKENS=32

# Stop any prior NRT on this port
pkill -f "nrt-server.*$NRT_ADDR" 2>/dev/null || true
sleep 0.5

PROMPTS=(
  "I was charged twice for my subscription last week"
  "My login page shows a 502 Bad Gateway error after the latest release"
  "Is a binding arbitration clause enforceable under California consumer law"
  "What is the capital of France"
)

echo "=== NRT vs Ollama runtime-only benchmark ==="
echo "Model:    TinyLlama-1.1B-Chat-v1.0 Q4_K_M (identical GGUF on both runtimes)"
echo "Prompts:  ${#PROMPTS[@]} × $MAX_TOKENS max tokens"
echo "Device:   NRT=Candle CPU, Ollama=Metal (M4)"
echo ""

# ---- 1. Boot NRT (Candle CPU) and warm ----
"$ROOT/target/release/nrt-server" \
  --manifest "$ROOT/manifests/customer-support.yaml" \
  --backend candle \
  --addr "$NRT_ADDR" \
  > /tmp/nrt-bench-server.log 2>&1 &
NRT_PID=$!
trap 'kill $NRT_PID 2>/dev/null || true' EXIT

for _ in {1..30}; do
  if curl -sf "http://$NRT_ADDR/healthz" >/dev/null 2>&1; then break; fi
  sleep 0.5
done
echo "NRT ready (pid $NRT_PID)"

# Warm NRT
curl -sf -X POST "http://$NRT_ADDR/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"warm"}],"max_tokens":2}' >/dev/null

# Warm Ollama
curl -sf -X POST "http://$OLLAMA_ADDR/api/generate" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$OLLAMA_MODEL\",\"prompt\":\"warm\",\"stream\":false,\"options\":{\"num_predict\":2,\"temperature\":0}}" \
  >/dev/null
echo "Both runtimes warmed."
echo ""

echo "--- NRT (Candle CPU, routed through router+specialist Manifest) ---"
NRT_TOTAL_MS=0
for i in "${!PROMPTS[@]}"; do
  PROMPT="${PROMPTS[$i]}"
  T0=$(python3 -c 'import time; print(int(time.time()*1000))')
  RESP=$(curl -sf -X POST "http://$NRT_ADDR/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$PROMPT")}],\"max_tokens\":$MAX_TOKENS}")
  T1=$(python3 -c 'import time; print(int(time.time()*1000))')
  LATENCY=$((T1 - T0))
  NRT_TOTAL_MS=$((NRT_TOTAL_MS + LATENCY))
  INTENT=$(echo "$RESP" | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r.get("nrt",{}).get("intent") or "-")')
  printf "  p%d intent=%-10s  %4d ms  %s\n" $((i+1)) "$INTENT" "$LATENCY" "$(echo "$PROMPT" | head -c 50)"
done
NRT_AVG_MS=$((NRT_TOTAL_MS / ${#PROMPTS[@]}))
echo "  avg total: $NRT_AVG_MS ms/request (router hop + specialist hop)"

echo ""
echo "--- Ollama (llama.cpp, Metal, single model) ---"
OLLAMA_TOTAL_MS=0
for i in "${!PROMPTS[@]}"; do
  PROMPT="${PROMPTS[$i]}"
  T0=$(python3 -c 'import time; print(int(time.time()*1000))')
  RESP=$(curl -sf -X POST "http://$OLLAMA_ADDR/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$OLLAMA_MODEL\",\"prompt\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$PROMPT"),\"stream\":false,\"options\":{\"num_predict\":$MAX_TOKENS,\"temperature\":0}}")
  T1=$(python3 -c 'import time; print(int(time.time()*1000))')
  LATENCY=$((T1 - T0))
  OLLAMA_TOTAL_MS=$((OLLAMA_TOTAL_MS + LATENCY))
  # Ollama returns total_duration in ns when stream=false.
  EVAL_COUNT=$(echo "$RESP" | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r.get("eval_count",0))')
  printf "  p%d eval_tokens=%-3d    %4d ms  %s\n" $((i+1)) "$EVAL_COUNT" "$LATENCY" "$(echo "$PROMPT" | head -c 50)"
done
OLLAMA_AVG_MS=$((OLLAMA_TOTAL_MS / ${#PROMPTS[@]}))
echo "  avg total: $OLLAMA_AVG_MS ms/request (single model, no dispatch)"

echo ""
echo "=== Summary ==="
printf "NRT    average latency (router + specialist, 2 forward passes): %d ms\n" "$NRT_AVG_MS"
printf "Ollama average latency (1 forward pass on Metal):               %d ms\n" "$OLLAMA_AVG_MS"
echo ""
echo "Honest read: NRT does TWO forward passes per request (router classifies"
echo "intent, then the specialist generates). Ollama does ONE. The per-hop NRT"
echo "number is what compares directly to Ollama's number."

# Cleanup
kill $NRT_PID 2>/dev/null || true
