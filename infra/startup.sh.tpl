#!/bin/bash
# Startup script: Docker + docker-compose 설치 + 데이터 디스크 마운트만 수행.
# 앱 코드 배포는 SSH로 직접 진행 (개발 단계에서 startup script에 박지 않음).
set -euo pipefail

# ── Docker 설치 ──────────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y ca-certificates curl gnupg lsb-release

mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io \
                   docker-buildx-plugin docker-compose-plugin

systemctl enable docker
systemctl start docker

# ── 데이터 디스크 포맷 및 마운트 ─────────────────────────────────────────────
# Terraform이 attached_disk device_name="statarb-data" 로 붙임.
DATA_DEV="/dev/disk/by-id/google-statarb-data"
MOUNT="/data"

mkdir -p "$MOUNT"

# 처음 attach 시에만 포맷 (이미 포맷됐으면 건너뜀).
if ! blkid "$DATA_DEV" | grep -q ext4; then
  mkfs.ext4 -F "$DATA_DEV"
fi

UUID=$(blkid -s UUID -o value "$DATA_DEV")

if ! grep -q "$MOUNT" /etc/fstab; then
  echo "UUID=$UUID $MOUNT ext4 discard,defaults,nofail 0 2" >> /etc/fstab
fi

mount -a

mkdir -p "$MOUNT/timescaledb"
chmod 777 "$MOUNT/timescaledb"

echo "[startup] Docker + data disk setup complete. SSH in to deploy app."
