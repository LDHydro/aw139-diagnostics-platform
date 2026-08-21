#!/usr/bin/env bash
# Launch the three vLLM servers on the RTX 5090, in the order that lets each
# size its KV cache correctly.
#
# Usage: deploy/start_models.sh [profile]      (default: balanced)
set -euo pipefail

PROFILE="${1:-balanced}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
CONFIG="$ROOT/config/models.yaml"
LOG_DIR="${ELP_LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

command -v vllm >/dev/null 2>&1 || {
  echo "error: vllm is not on PATH. Activate the environment created by deploy/install_host.sh." >&2
  exit 1
}

read_profile() {
  python3 - "$CONFIG" "$PROFILE" "$1" "$2" <<'PY'
import sys, yaml
config, profile, section, key = sys.argv[1:5]
data = yaml.safe_load(open(config))
try:
    node = data["profiles"][profile][section]
except KeyError:
    sys.exit(f"no such profile/section: {profile}/{section}")
value = node.get(key, "")
if isinstance(value, list):
    print(" ".join(str(v) for v in value))
else:
    print(value)
PY
}

wait_for() {
  local url="$1" name="$2" tries="${3:-120}"
  echo -n "  waiting for $name "
  for _ in $(seq 1 "$tries"); do
    if curl -sf "$url" >/dev/null 2>&1; then echo " ready"; return 0; fi
    echo -n "."
    sleep 5
  done
  echo " TIMED OUT"
  echo "  check $LOG_DIR/$name.log" >&2
  return 1
}

launch() {
  local name="$1" section="$2" extra_task="$3"
  local model port util maxlen quant kvdtype extra
  model=$(read_profile "$section" model)
  port=$(read_profile "$section" port)
  util=$(read_profile "$section" gpu_memory_utilization)
  maxlen=$(read_profile "$section" max_model_len)
  quant=$(read_profile "$section" quantization)
  kvdtype=$(read_profile "$section" kv_cache_dtype)
  extra=$(read_profile "$section" extra_args)

  local -a args=(
    serve "$model"
    --host 127.0.0.1
    --port "$port"
    --served-model-name "$model"
    --gpu-memory-utilization "$util"
    --max-model-len "$maxlen"
    --disable-log-requests
  )
  [[ -n "$quant" ]]      && args+=(--quantization "$quant")
  [[ -n "$kvdtype" ]]    && args+=(--kv-cache-dtype "$kvdtype")
  [[ -n "$extra_task" ]] && args+=(--task "$extra_task")
  # shellcheck disable=SC2206
  [[ -n "$extra" ]] && args+=($extra)

  echo "starting $name: $model on :$port (util $util)"
  nohup vllm "${args[@]}" > "$LOG_DIR/$name.log" 2>&1 &
  echo "$!" > "$LOG_DIR/$name.pid"
}

echo "profile: $PROFILE"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version \
           --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"

# Small servers first: vLLM sizes its KV cache from free memory at start-up,
# so the large model must be the one that measures last.
launch embeddings embeddings embed
wait_for "http://127.0.0.1:$(read_profile embeddings port)/health" embeddings 60

launch reranker reranker score
wait_for "http://127.0.0.1:$(read_profile reranker port)/health" reranker 60

launch chat chat ""
# The chat model is large and crosses the external GPU link once at load;
# several minutes on first start is normal.
wait_for "http://127.0.0.1:$(read_profile chat port)/health" chat 240

echo
echo "all model servers are up. VRAM in use:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
