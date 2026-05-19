#!/usr/bin/env sh
set -eu

mkdir -p /logs /output
python /app/main.py 2>&1 | tee /logs/runtime.log
