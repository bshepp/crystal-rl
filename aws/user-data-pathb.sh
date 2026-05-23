#!/bin/bash
# Path-B retrain pipeline: 16-element fingerprint palette + Sb/Bi/Se/Te DFT support.
# Sequence: regenerate MP fingerprints (156-dim) → retrain surrogate → retrain PPO →
# DFT-validate top candidates → upload all artifacts → self-terminate.
set -ex
exec > >(tee /var/log/pathb-run.log) 2>&1

BUCKET="rl-materials-bootstrap-290318879194"
RUN_ID="pathb-$(date +%Y%m%d-%H%M%S)"
S3_BASE="s3://${BUCKET}/runs/${RUN_ID}"
echo "=== Path-B run starting at $(date) ==="
echo "RUN_ID=${RUN_ID}"
echo "S3_BASE=${S3_BASE}"

# ---- Status helper: write a file locally and to S3 so we can poll ----
status() {
  echo "[STATUS] $1 at $(date)"
  echo "$1" > /tmp/status.txt
  aws s3 cp /tmp/status.txt "${S3_BASE}/status.txt" || true
}
status "starting"

# ---- Install Docker ----
dnf install -y docker
systemctl start docker
systemctl enable docker
status "docker-installed"

# ---- Fetch source ----
mkdir -p /opt/rl-materials
cd /opt/rl-materials
aws s3 cp "s3://${BUCKET}/src/rl-materials-src-pathb.tar.gz" .
tar xzf rl-materials-src-pathb.tar.gz
rm rl-materials-src-pathb.tar.gz
status "src-extracted"

# ---- Fetch MP API key from SSM (encrypted in transit + at rest) ----
MP_API_KEY=$(aws ssm get-parameter \
  --name "/rl-materials/mp-api-key" \
  --with-decryption \
  --region us-east-1 \
  --query Parameter.Value --output text)
if [ -z "$MP_API_KEY" ]; then
  echo "FATAL: could not fetch MP API key from SSM"
  status "FAILED-ssm-fetch"
  shutdown -h now
fi
status "ssm-key-fetched"

# ---- Build Docker image ----
echo "=== Building Docker image at $(date) ==="
docker build -t rl-materials-qe-rl:latest .
echo "=== Docker build complete at $(date) ==="
status "docker-built"

# ---- Mount points ----
mkdir -p /opt/results /opt/data
# Pre-populate data/ with the raw bootstrap JSONs from the tarball
cp -r /opt/rl-materials/data/* /opt/data/ 2>/dev/null || true

# Common Docker run flags (used by every stage)
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

# =====================================================================
# Stage 1: regenerate MP fingerprints with the new 16-element palette
# =====================================================================
echo "=== Stage 1: MP fingerprint regen at $(date) ==="
status "stage1-mp-fingerprints"
# Delete any stale (152-dim) MP cache that snuck into the source tarball
rm -f /opt/data/mp_fingerprints.npz /opt/data/jarvis_fingerprints.npz \
      /opt/data/jarvis_gap_only_fingerprints.npz
docker run "${DOCKER_FLAGS[@]}" rl-materials-qe-rl:latest \
  python3 -m scripts.download_mp --api-key "$MP_API_KEY" --max-atoms 50 \
    --cache-file data/mp_fingerprints.npz 2>&1 | tee /var/log/stage1.log
aws s3 cp /opt/data/mp_fingerprints.npz "${S3_BASE}/mp_fingerprints.npz" || true
aws s3 cp /var/log/stage1.log "${S3_BASE}/logs/stage1-mp.log" || true
status "stage1-done"

# =====================================================================
# Stage 2: surrogate retrain (two-phase, JARVIS + MP, 156-dim)
# =====================================================================
echo "=== Stage 2: surrogate retrain at $(date) ==="
status "stage2-surrogate"
docker run "${DOCKER_FLAGS[@]}" rl-materials-qe-rl:latest \
  python3 -m scripts.retrain_jarvis \
    --include-mp --max-mp 4000 \
    --hidden-dim 192 --gap-weight 0.3 --two-phase \
    --save-dir data/checkpoints/jarvis_surrogate 2>&1 | tee /var/log/stage2.log
aws s3 sync /opt/data/checkpoints/jarvis_surrogate \
  "${S3_BASE}/checkpoints/jarvis_surrogate/" || true
aws s3 cp /opt/data/jarvis_fingerprints.npz \
  "${S3_BASE}/jarvis_fingerprints.npz" || true
aws s3 cp /var/log/stage2.log "${S3_BASE}/logs/stage2-surrogate.log" || true
status "stage2-done"

# =====================================================================
# Stage 3: PPO retrain against new surrogate
# =====================================================================
echo "=== Stage 3: PPO retrain at $(date) ==="
status "stage3-ppo"
docker run "${DOCKER_FLAGS[@]}" rl-materials-qe-rl:latest \
  python3 -m scripts.train_ppo_jarvis --timesteps 250000 2>&1 | tee /var/log/stage3.log
aws s3 sync /opt/data/checkpoints/ppo_jarvis \
  "${S3_BASE}/checkpoints/ppo_jarvis/" || true
aws s3 cp /var/log/stage3.log "${S3_BASE}/logs/stage3-ppo.log" || true
status "stage3-done"

# =====================================================================
# Stage 4: DFT validation of top candidates
# =====================================================================
echo "=== Stage 4: DFT validation at $(date) ==="
status "stage4-dft"
# QE benefits from multiple processes here
docker run "${DOCKER_FLAGS[@]}" \
  -e OMP_NUM_THREADS=4 \
  rl-materials-qe-rl:latest \
  python3 -m scripts.validate_dft 2>&1 | tee /var/log/stage4.log
aws s3 sync /opt/data/validation "${S3_BASE}/validation/" || true
aws s3 cp /var/log/stage4.log "${S3_BASE}/logs/stage4-dft.log" || true
status "stage4-done"

# =====================================================================
# Final sync + self-terminate
# =====================================================================
aws s3 cp /var/log/pathb-run.log "${S3_BASE}/logs/full-run.log" || true
aws s3 sync /opt/data "${S3_BASE}/data/" \
  --exclude "*.tar.gz" \
  --exclude "bootstrap/*" || true
status "COMPLETE"
echo "=== Path-B run complete at $(date) ==="

# Brief grace period for final log line to flush, then self-terminate
sleep 30
shutdown -h now
