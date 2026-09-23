#!/bin/sh
set -eu

lock_hash=$(sha256sum /workspace/requirements-lock.txt | cut -d ' ' -f 1)
if [ ! -x /opt/venv/bin/python ] || [ ! -f /opt/venv/.qsarify-lock-hash ] || [ "$(cat /opt/venv/.qsarify-lock-hash)" != "$lock_hash" ]; then
  python -m venv /opt/venv
  sed '/^xgboost==/d' /workspace/requirements-lock.txt > /tmp/requirements-runtime.txt
  /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements-runtime.txt
  /opt/venv/bin/pip install --no-cache-dir --no-deps xgboost==3.2.0
  rm -f /tmp/requirements-runtime.txt
  printf '%s' "$lock_hash" > /opt/venv/.qsarify-lock-hash
fi

exec /opt/venv/bin/"$@"
