#!/bin/bash
set -euo pipefail

apt-get update -y
apt-get install -y python3 python3-pip python3-venv

mkdir -p /opt/collector
gsutil cp gs://${bucket_name}/main.py /opt/collector/
gsutil cp gs://${bucket_name}/requirements.txt /opt/collector/

python3 -m venv /opt/collector/venv
/opt/collector/venv/bin/pip install --quiet -r /opt/collector/requirements.txt

BINANCE_API_KEY=$(gcloud secrets versions access latest \
  --secret=binance-api-key --project=${project_id})
BINANCE_API_SECRET=$(gcloud secrets versions access latest \
  --secret=binance-api-secret --project=${project_id})

cat > /opt/collector/.env << ENVEOF
BINANCE_API_KEY="$BINANCE_API_KEY"
BINANCE_API_SECRET="$BINANCE_API_SECRET"
BQ_PROJECT="${project_id}"
ENVEOF

cat > /etc/systemd/system/collector.service << 'SVCEOF'
[Unit]
Description=Crypto OHLCV Collector
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/collector
EnvironmentFile=/opt/collector/.env
ExecStart=/opt/collector/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable collector
systemctl start collector
