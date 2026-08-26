#!/usr/bin/env bash
# Stop the vLLM servers started by start_models.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ELP_LOG_DIR:-$(dirname "$HERE")/logs}"

for name in chat reranker embeddings; do
  pidfile="$LOG_DIR/$name.pid"
  [[ -f "$pidfile" ]] || continue
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "stopping $name (pid $pid)"
    kill "$pid"
    # vLLM needs a moment to release VRAM cleanly; a hard kill can leave
    # the card holding memory until the driver resets it.
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pidfile"
done
echo "done."
