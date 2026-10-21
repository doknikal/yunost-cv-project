#!/bin/bash
# Safe app-only installation. The notebook .venv is never touched.
set -e
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
for PY in "${YUNOST_PYTHON:-}" python3.12 python3.11 python3.10 python3.9 /usr/bin/python3 python3; do
  if [ -n "$PY" ] && command -v "$PY" >/dev/null 2>&1 && "$PY" -c 'import sys; sys.exit(not ((3,9) <= sys.version_info[:2] <= (3,12)))' 2>/dev/null; then
    "$PY" setup_app.py
    echo "Установка завершена. Откройте start_mac.command."
    exit 0
  fi
done
echo "Нужен Python 3.9–3.12: brew install python@3.12 tesseract"
exit 1
