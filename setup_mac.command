#!/bin/bash
# Двойной клик настроит окружение для ноутбука на Homebrew-Python (/opt/homebrew/bin/python3).
# Пересоздаёт .venv на этом интерпретаторе, ставит библиотеки, регистрирует ядро Jupyter
# и регенерирует data/norm из соседней папки ../1967-12.
set -e
cd "$(dirname "$0")"
echo "==> Папка проекта: $(pwd)"

PY="/opt/homebrew/bin/python3"
if [ ! -x "$PY" ]; then
  echo "!! $PY не найден. Установи Homebrew-Python: brew install python"; exit 1
fi
echo "==> Интерпретатор: $PY ($($PY --version 2>&1))"

# деактивировать и пересоздать .venv именно на Homebrew-Python
deactivate 2>/dev/null || true
if [ -d ".venv" ]; then echo "==> Удаляю старый .venv ..."; rm -rf .venv; fi
echo "==> Создаю .venv на Homebrew-Python ..."
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel >/dev/null

echo "==> Ставлю библиотеки (есть готовые wheels под 3.14, минута-две)..."
pip install -r requirements.txt

echo "==> Регистрирую ядро Jupyter 'Python (Yunost CV)'..."
python -m ipykernel install --user --name yunost-cv --display-name "Python (Yunost CV)"

if [ -d "../1967-12" ]; then
  echo "==> Регенерирую нормализованные страницы из ../1967-12 ..."
  python prep_package.py || echo "(предупреждение: prep пропущен)"
else
  echo "==> Папка ../1967-12 рядом не найдена — пропускаю prep (результаты в ноутбуке уже встроены)."
fi

echo ""
echo "ГОТОВО ✅  Интерпретатор: $(python --version) на базе Homebrew."
echo "В VS Code: открой ноутбук -> 'Select Kernel' -> выбери '.venv' (Python 3.14.6) -> Run All."
