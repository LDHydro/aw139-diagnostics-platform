#!/usr/bin/env bash
# Prepare a workstation to run the Enterprise LLM Platform.
#
# Targets a Linux host with an NVIDIA RTX 5090. The 5090 is Blackwell
# (sm_120), which needs a 570-series or newer driver and a CUDA 12.8+ build
# of PyTorch. Wheels built for earlier architectures will install happily and
# then fail at runtime with "no kernel image is available for execution on
# the device" - so this script checks before it installs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
VENV="${ELP_VENV:-$ROOT/.venv}"
PYTHON="${ELP_PYTHON:-python3}"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m warn:\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------
info "Host"
# ---------------------------------------------------------------------
# Read the CPU rather than assuming it: Threadripper and Ryzen 9 are
# different product lines with different core and NUMA layouts, and the
# right thread count depends on which one is actually fitted.
CPU_MODEL="$(grep -m1 '^model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//' || echo unknown)"
CORES="$(nproc)"
RAM_GB="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)"
echo "  CPU:  $CPU_MODEL"
echo "  Cores: $CORES     RAM: ${RAM_GB} GB"
[[ "$RAM_GB" -lt 32 ]] && warn "under 32 GB of RAM; document ingestion may struggle"

# ---------------------------------------------------------------------
info "GPU"
# ---------------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || fail \
  "nvidia-smi not found. Install the NVIDIA driver (570 or newer) first."

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
DRIVER_MAJOR="${DRIVER%%.*}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"

if [[ "$DRIVER_MAJOR" -lt 570 ]]; then
  fail "driver $DRIVER is too old for Blackwell. The RTX 5090 needs 570 or newer."
fi
if [[ "$VRAM_MB" -lt 30000 ]]; then
  warn "only ${VRAM_MB} MB of VRAM detected. The default 'balanced' profile assumes 32 GB;
       use the 'long_context' or 'throughput' profile, or a smaller model."
fi

# An external GPU sits on a much narrower link than a desktop slot. This is
# fine for inference but makes model loading slow, so it is worth knowing.
LINK_WIDTH="$(nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '')"
if [[ -n "$LINK_WIDTH" && "$LINK_WIDTH" -le 4 ]]; then
  warn "PCIe link is x${LINK_WIDTH} - consistent with an external GPU enclosure.
       Model loading will take minutes. Inference speed is unaffected because
       the weights stay resident in VRAM. Do NOT enable CPU offload or swap."
fi

# ---------------------------------------------------------------------
info "Python environment"
# ---------------------------------------------------------------------
PY_VERSION="$($PYTHON -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PY_VERSION" in
  3.11|3.12) ;;
  *) warn "Python $PY_VERSION detected; 3.11 or 3.12 is recommended for vLLM." ;;
esac

if [[ ! -d "$VENV" ]]; then
  info "creating virtual environment at $VENV"
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools >/dev/null

# ---------------------------------------------------------------------
info "PyTorch (CUDA 12.8, required for Blackwell)"
# ---------------------------------------------------------------------
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

info "verifying the GPU is actually usable from PyTorch"
python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("PyTorch cannot see the GPU. Check the driver and that the eGPU is attached.")

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"  device: {name}  (sm_{major}{minor})  torch {torch.__version__}")

supported = torch.cuda.get_arch_list()
print(f"  wheel supports: {', '.join(supported)}")
if f"sm_{major}{minor}" not in supported:
    sys.exit(
        f"This PyTorch build has no kernels for sm_{major}{minor}. "
        "Reinstall from the cu128 index."
    )

# Prove it end to end - arch lists can lie about what actually runs.
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
torch.matmul(x, x)
torch.cuda.synchronize()
print("  matmul on GPU: OK")
PY

# ---------------------------------------------------------------------
info "vLLM and platform dependencies"
# ---------------------------------------------------------------------
pip install "vllm>=0.9.0"
pip install -e "$ROOT"

# ---------------------------------------------------------------------
info "PostgreSQL with pgvector"
# ---------------------------------------------------------------------
if command -v psql >/dev/null 2>&1; then
  echo "  psql found: $(psql --version)"
else
  warn "psql not found. Either install PostgreSQL 16+ with pgvector, or run
       'docker compose up -d postgres' from $ROOT to use the bundled container."
fi

# ---------------------------------------------------------------------
info "LaTeX"
# ---------------------------------------------------------------------
if command -v tectonic >/dev/null 2>&1; then
  echo "  tectonic found: $(tectonic --version 2>&1 | head -1)"
elif command -v latexmk >/dev/null 2>&1; then
  echo "  latexmk found; set ELP_LATEX__ENGINE=latexmk"
else
  warn "no LaTeX engine found. Install tectonic (recommended, self-contained)
       or texlive-full, or leave ELP_LATEX__ENABLED=false."
fi

cat <<EOF

$(info "Next steps")
  1. cp .env.example .env    and fill in the database URL and your SSO settings
  2. docker compose up -d postgres          (unless you run your own)
  3. deploy/start_models.sh balanced        (first load takes several minutes)
  4. python scripts/bootstrap.py            (creates the schema and an admin key)
  5. uvicorn elp.main:app --host 0.0.0.0 --port 8080

  GPU: $GPU_NAME, driver $DRIVER, ${VRAM_MB} MB VRAM
  Virtual environment: $VENV
EOF
