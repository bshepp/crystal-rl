#!/bin/bash
# Path-B retrain pipeline: 16-element fingerprint palette + Sb/Bi/Se/Te DFT.
#
# Sequence:
#   stage1: regenerate MP fingerprints (156-dim)
#   stage2: retrain surrogate (two-phase, JARVIS + MP)
#   stage3: retrain PPO (250k timesteps)
#   stage4: DFT-validate top candidates
# Each stage is idempotent: its output artifact is mirrored to S3, and on
# relaunch the script will skip stages whose output already exists in S3
# (with a size sanity check). On failure mid-pipeline, terminate without
# losing prior stage output — the next launch resumes from the failed stage.
#
# RUN_ID resolution (in order):
#   1. RUN_ID env var (if passed via launch parameter)
#   2. The most recent S3 run with status != COMPLETE (auto-resume)
#   3. A fresh pathb-<timestamp> RUN_ID

set -eo pipefail
exec > >(tee /var/log/pathb-run.log) 2>&1

BUCKET="rl-materials-bootstrap-290318879194"
REGION="us-east-1"

echo "=== Path-B run launching at $(date) ==="

# ----------------------------------------------------------------------
# Install AWS CLI deps (need it for the resume probe before anything else)
# ----------------------------------------------------------------------
if ! command -v aws &>/dev/null; then
  dnf install -y awscli
fi

# ----------------------------------------------------------------------
# RUN_ID resolution (resume vs fresh)
# ----------------------------------------------------------------------
resolve_run_id() {
  if [ -n "${RUN_ID:-}" ]; then
    echo "Using RUN_ID from environment: $RUN_ID"
    return
  fi
  # Probe S3 for most recent run that is not COMPLETE.
  # NB: every pipeline element here is `|| true`-guarded because we run with
  # `set -eo pipefail` and an empty `grep` exits 1, which would abort the
  # whole script the very first time (when runs/ is empty).
  local latest
  latest=$( { aws s3 ls "s3://${BUCKET}/runs/" --region "$REGION" 2>/dev/null \
    | awk '{print $2}' \
    | grep '^pathb-' \
    | sort -r \
    | head -5 ; } || true )
  for candidate in $latest; do
    candidate=${candidate%/}
    local prior_status
    prior_status=$(aws s3 cp "s3://${BUCKET}/runs/${candidate}/status.txt" - \
      --region "$REGION" 2>/dev/null || true)
    if [ -n "$prior_status" ] && [ "$prior_status" != "COMPLETE" ]; then
      echo "Resuming incomplete prior run: ${candidate} (last status: ${prior_status})"
      RUN_ID="$candidate"
      return
    fi
  done
  RUN_ID="pathb-$(date -u +%Y%m%d-%H%M%S)"
  echo "Starting fresh run: $RUN_ID"
}
resolve_run_id

S3_BASE="s3://${BUCKET}/runs/${RUN_ID}"
echo "RUN_ID=${RUN_ID}"
echo "S3_BASE=${S3_BASE}"

# ----------------------------------------------------------------------
# Status helper: write a status file locally and to S3 so we can poll
# ----------------------------------------------------------------------
status() {
  echo "[STATUS $(date -u +%H:%M:%S)] $1"
  echo "$1" > /tmp/status.txt
  aws s3 cp /tmp/status.txt "${S3_BASE}/status.txt" --region "$REGION" \
    >/dev/null 2>&1 || true
}
status "starting"

# ----------------------------------------------------------------------
# Failure trap: mark status FAILED-<last-stage> and DON'T self-terminate
# (instance stays up so we can SSM in and inspect; manual termination
# required). For automatic termination on success, see end of script.
# ----------------------------------------------------------------------
LAST_STAGE="init"
on_failure() {
  local rc=$?
  status "FAILED-${LAST_STAGE} (rc=${rc})"
  aws s3 cp /var/log/pathb-run.log "${S3_BASE}/logs/full-run-failed.log" \
    --region "$REGION" >/dev/null 2>&1 || true
  echo "=== FAILED at stage ${LAST_STAGE} (rc=${rc}); instance left running for inspection ==="
  exit "$rc"
}
trap on_failure ERR

# ----------------------------------------------------------------------
# Idempotency helpers: does a stage's S3 artifact already exist?
# We use file size as a coarse validity check (rejects 0-byte uploads).
# ----------------------------------------------------------------------
s3_artifact_exists() {
  # $1: S3 key path under S3_BASE
  # $2: minimum acceptable size in bytes (e.g., 1000 for a small JSON, 10000 for npz)
  local key="$1"
  local min_size="$2"
  local size
  size=$(aws s3api head-object \
    --bucket "${BUCKET}" \
    --key "runs/${RUN_ID}/${key}" \
    --region "$REGION" \
    --query 'ContentLength' \
    --output text 2>/dev/null || echo "0")
  if [ -n "$size" ] && [ "$size" != "None" ] && [ "$size" -ge "$min_size" ]; then
    echo "  [CACHED] ${key} (${size} bytes) — skipping stage"
    return 0
  else
    return 1
  fi
}

s3_download() {
  # $1: S3 key under S3_BASE  $2: local target path
  local key="$1"
  local target="$2"
  mkdir -p "$(dirname "$target")"
  aws s3 cp "${S3_BASE}/${key}" "$target" --region "$REGION"
}

s3_upload() {
  # $1: local source  $2: S3 key under S3_BASE
  local source="$1"
  local key="$2"
  if [ -e "$source" ]; then
    aws s3 cp "$source" "${S3_BASE}/${key}" --region "$REGION" \
      >/dev/null 2>&1 || true
  fi
}

s3_sync_up() {
  # $1: local dir  $2: S3 prefix under S3_BASE
  local source="$1"
  local prefix="$2"
  if [ -d "$source" ]; then
    aws s3 sync "$source" "${S3_BASE}/${prefix}" --region "$REGION" \
      >/dev/null 2>&1 || true
  fi
}

s3_sync_down() {
  # $1: S3 prefix under S3_BASE  $2: local dir
  local prefix="$1"
  local target="$2"
  mkdir -p "$target"
  aws s3 sync "${S3_BASE}/${prefix}" "$target" --region "$REGION" \
    >/dev/null 2>&1 || true
}

# ----------------------------------------------------------------------
# Install Docker (idempotent — dnf is OK on re-runs)
# ----------------------------------------------------------------------
LAST_STAGE="install-docker"
if ! command -v docker &>/dev/null; then
  dnf install -y docker
  systemctl start docker
  systemctl enable docker
fi
status "docker-installed"

# ----------------------------------------------------------------------
# Fetch source (always overwrite — we want the canonical bundle)
# ----------------------------------------------------------------------
LAST_STAGE="fetch-source"
mkdir -p /opt/rl-materials
cd /opt/rl-materials
rm -rf ./*
aws s3 cp "s3://${BUCKET}/src/rl-materials-src-pathb.tar.gz" . --region "$REGION"
tar xzf rl-materials-src-pathb.tar.gz
rm rl-materials-src-pathb.tar.gz
status "src-extracted"

# ----------------------------------------------------------------------
# Fetch MP API key from SSM (encrypted in transit + at rest)
# ----------------------------------------------------------------------
LAST_STAGE="fetch-ssm-key"
MP_API_KEY=$(aws ssm get-parameter \
  --name "/rl-materials/mp-api-key" \
  --with-decryption \
  --region "$REGION" \
  --query Parameter.Value --output text)
if [ -z "$MP_API_KEY" ]; then
  echo "FATAL: could not fetch MP API key from SSM"
  exit 1
fi
status "ssm-key-fetched"

# ----------------------------------------------------------------------
# Build Docker image (idempotent — Docker layer cache makes repeats fast)
# ----------------------------------------------------------------------
LAST_STAGE="docker-build"
if ! docker image inspect rl-materials-qe-rl:latest >/dev/null 2>&1; then
  echo "=== Building Docker image at $(date) ==="
  docker build -t rl-materials-qe-rl:latest .
  echo "=== Docker build complete at $(date) ==="
else
  echo "Docker image already present, skipping build"
fi
status "docker-built"

# ----------------------------------------------------------------------
# Mount points (idempotent)
# ----------------------------------------------------------------------
mkdir -p /opt/results /opt/data /opt/data/checkpoints /opt/data/validation
# Pre-populate data/ with raw bootstrap JSONs from the source tarball
cp -rn /opt/rl-materials/data/* /opt/data/ 2>/dev/null || true

DOCKER_FLAGS=(
  --rm
  -v /opt/data:/workspace/data
  -v /opt/results:/workspace/results
  -e OMP_NUM_THREADS=1
  -e OMPI_ALLOW_RUN_AS_ROOT=1
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -e PYTHONPATH=/workspace
  -e MP_API_KEY="$MP_API_KEY"
)

# ======================================================================
# Stage 1: regenerate MP fingerprints (156-dim, 16-element palette)
# ======================================================================
LAST_STAGE="stage1-mp-fingerprints"
echo "=== Stage 1: MP fingerprint regen at $(date) ==="
if s3_artifact_exists "mp_fingerprints.npz" 100000; then
  s3_download "mp_fingerprints.npz" /opt/data/mp_fingerprints.npz
  status "stage1-done (resumed from S3)"
else
  status "stage1-running"
  # Clear any stale dim-mismatched caches inherited from the source tarball
  rm -f /opt/data/mp_fingerprints.npz \
        /opt/data/jarvis_fingerprints.npz \
        /opt/data/jarvis_gap_only_fingerprints.npz
  docker run "${DOCKER_FLAGS[@]}" rl-materials-qe-rl:latest \
    python3 -m scripts.download_mp \
      --api-key "$MP_API_KEY" --max-atoms 50 \
      --cache-file data/mp_fingerprints.npz \
    2>&1 | tee /var/log/stage1.log
  # Verify output is non-trivial before declaring success
  if [ ! -s /opt/data/mp_fingerprints.npz ] || \
     [ "$(stat -c%s /opt/data/mp_fingerprints.npz)" -lt 100000 ]; then
    echo "ERROR: stage1 produced no/tiny mp_fingerprints.npz"
    exit 1
  fi
  s3_upload /opt/data/mp_fingerprints.npz "mp_fingerprints.npz"
  s3_upload /var/log/stage1.log "logs/stage1-mp.log"
  status "stage1-done"
fi

# ======================================================================
# Stage 2: surrogate retrain (two-phase, JARVIS + MP, 156-dim)
# ======================================================================
LAST_STAGE="stage2-surrogate"
echo "=== Stage 2: surrogate retrain at $(date) ==="
if s3_artifact_exists "checkpoints/jarvis_surrogate/surrogate_weights.pt" 50000 && \
   s3_artifact_exists "checkpoints/jarvis_surrogate/surrogate_norm.npz" 200; then
  s3_sync_down "checkpoints/jarvis_surrogate/" /opt/data/checkpoints/jarvis_surrogate/
  s3_download "jarvis_fingerprints.npz" /opt/data/jarvis_fingerprints.npz \
    2>/dev/null || true
  status "stage2-done (resumed from S3)"
else
  status "stage2-running"
  docker run "${DOCKER_FLAGS[@]}" rl-materials-qe-rl:latest \
    python3 -m scripts.retrain_jarvis \
      --include-bootstrap \
      --include-mp --max-mp 4000 \
      --hidden-dim 192 --gap-weight 0.3 --two-phase \
      --save-dir data/checkpoints/jarvis_surrogate \
    2>&1 | tee /var/log/stage2.log
  # Verify the surrogate weights were saved
  if [ ! -s /opt/data/checkpoints/jarvis_surrogate/surrogate_weights.pt ]; then
    echo "ERROR: stage2 produced no surrogate_weights.pt"
    exit 1
  fi
  s3_sync_up /opt/data/checkpoints/jarvis_surrogate "checkpoints/jarvis_surrogate/"
  s3_upload /opt/data/jarvis_fingerprints.npz "jarvis_fingerprints.npz"
  s3_upload /var/log/stage2.log "logs/stage2-surrogate.log"
  status "stage2-done"
fi

# ======================================================================
# Stage 3: PPO retrain against new surrogate
# ======================================================================
LAST_STAGE="stage3-ppo"
echo "=== Stage 3: PPO retrain at $(date) ==="
if s3_artifact_exists "checkpoints/ppo_jarvis/ppo_final.zip" 100000; then
  s3_sync_down "checkpoints/ppo_jarvis/" /opt/data/checkpoints/ppo_jarvis/
  status "stage3-done (resumed from S3)"
else
  status "stage3-running"
  docker run "${DOCKER_FLAGS[@]}" rl-materials-qe-rl:latest \
    python3 -m scripts.train_ppo_jarvis --timesteps 250000 \
    2>&1 | tee /var/log/stage3.log
  if [ ! -s /opt/data/checkpoints/ppo_jarvis/ppo_final.zip ]; then
    echo "ERROR: stage3 produced no ppo_final.zip"
    exit 1
  fi
  s3_sync_up /opt/data/checkpoints/ppo_jarvis "checkpoints/ppo_jarvis/"
  s3_upload /var/log/stage3.log "logs/stage3-ppo.log"
  status "stage3-done"
fi

# ======================================================================
# Stage 4: DFT validation of top candidates
# ======================================================================
LAST_STAGE="stage4-dft"
echo "=== Stage 4: DFT validation at $(date) ==="
if s3_artifact_exists "validation/validation_report.json" 500; then
  s3_sync_down "validation/" /opt/data/validation/
  status "stage4-done (resumed from S3)"
else
  status "stage4-running"
  docker run "${DOCKER_FLAGS[@]}" \
    -e OMP_NUM_THREADS=4 \
    rl-materials-qe-rl:latest \
    python3 -m scripts.validate_dft \
    2>&1 | tee /var/log/stage4.log
  if [ ! -s /opt/data/validation/validation_report.json ]; then
    echo "ERROR: stage4 produced no validation_report.json"
    exit 1
  fi
  s3_sync_up /opt/data/validation "validation/"
  s3_upload /var/log/stage4.log "logs/stage4-dft.log"
  status "stage4-done"
fi

# ======================================================================
# Final: write a results manifest, sync everything, self-terminate
# ======================================================================
LAST_STAGE="finalize"
echo "=== Finalizing at $(date) ==="

# Generate a manifest of all artifacts for easy download
{
  echo "{"
  echo "  \"run_id\": \"${RUN_ID}\","
  echo "  \"completed_at\": \"$(date -u --iso-8601=seconds)\","
  echo "  \"s3_base\": \"${S3_BASE}\","
  echo "  \"artifacts\": {"
  echo "    \"mp_fingerprints\": \"${S3_BASE}/mp_fingerprints.npz\","
  echo "    \"jarvis_fingerprints\": \"${S3_BASE}/jarvis_fingerprints.npz\","
  echo "    \"surrogate_dir\": \"${S3_BASE}/checkpoints/jarvis_surrogate/\","
  echo "    \"ppo_dir\": \"${S3_BASE}/checkpoints/ppo_jarvis/\","
  echo "    \"validation_report\": \"${S3_BASE}/validation/validation_report.json\","
  echo "    \"unusual_topology\": \"${S3_BASE}/validation/unusual_topology.json\","
  echo "    \"full_log\": \"${S3_BASE}/logs/full-run.log\""
  echo "  }"
  echo "}"
} > /tmp/manifest.json
aws s3 cp /tmp/manifest.json "${S3_BASE}/manifest.json" --region "$REGION"

# Final blanket sync (catches anything the stage uploaders missed)
aws s3 cp /var/log/pathb-run.log "${S3_BASE}/logs/full-run.log" \
  --region "$REGION" || true
aws s3 sync /opt/data "${S3_BASE}/data/" \
  --region "$REGION" \
  --exclude "*.tar.gz" \
  --exclude "bootstrap/s3-full/*" || true

status "COMPLETE"
echo "=== Path-B run COMPLETE at $(date) ==="
echo "All artifacts at: ${S3_BASE}/"
echo "Manifest:         ${S3_BASE}/manifest.json"

# Grace period for the final log line to flush, then self-terminate.
# (Instance has --instance-initiated-shutdown-behavior terminate, so this
# actually terminates the instance and releases the EBS volume.)
sleep 30
shutdown -h now
