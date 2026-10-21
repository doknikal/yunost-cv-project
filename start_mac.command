#!/bin/bash
set -e
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
if [ ! -x .venv-app/bin/python ]; then
  echo "Сначала запустите install_mac.command."
  exit 1
fi
exec .venv-app/bin/python run_app.py "$@"
