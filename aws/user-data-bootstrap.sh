#!/bin/bash
set -ex

# Log everything for debugging
exec > >(tee /var/log/bootstrap-run.log) 2>&1
echo "=== rl-materials expanded bootstrap starting at $(date) ==="

# ---- Install Docker ----
dnf install docker -y
systemctl start docker
systemctl enable docker

# ---- Download project from S3 ----
mkdir -p /opt/rl-materials
cd /opt/rl-materials
aws s3 cp s3://rl-materials-bootstrap-290318879194/src/rl-materials-src.tar.gz .
tar xzf rl-materials-src.tar.gz
rm rl-materials-src.tar.gz

# ---- Build Docker image ----
echo "=== Building Docker image at $(date) ==="
docker build -t rl-materials-qe-rl:latest .
echo "=== Docker build complete at $(date) ==="

# ---- Create output directory ----
mkdir -p /opt/results

# ---- Run expanded bootstrap ----
# 4 rounds x ~200 structures = ~800 data points
# 8 workers x 2 MPI procs = 16 vCPU fully utilised
echo "=== Starting expanded bootstrap (4 rounds) at $(date) ==="
docker run \
  --rm \
  -v /opt/results:/workspace/data \
  -e OMP_NUM_THREADS=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  rl-materials-qe-rl:latest \
  python3 -m scripts.bootstrap_expanded \
    --rounds 4 \
    --workers 8 \
    --np-procs 2 \
    --seed 2024 \
    --output-dir data/bootstrap

echo "=== Bootstrap complete at $(date) ==="

# ---- Upload results to S3 ----
aws s3 sync /opt/results s3://rl-materials-bootstrap-290318879194/results/

# ---- Upload log ----
aws s3 cp /var/log/bootstrap-run.log s3://rl-materials-bootstrap-290318879194/logs/bootstrap-run.log

echo "=== All done, shutting down at $(date) ==="

# ---- Self-terminate ----
shutdown -h now
