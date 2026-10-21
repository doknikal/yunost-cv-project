"""Local-only launcher with dependency/model diagnostics."""
import argparse
import os
from pathlib import Path
import socket
import sys
import threading
import webbrowser

from setup_app import check_tesseract, verify_weights

HERE = Path(__file__).resolve().parent


def check_port(port):
    with socket.socket() as probe:
        # Match uvicorn's bind semantics; TIME_WAIT is not a running server.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', port))
        except OSError as exc:
            raise RuntimeError(f'Порт {port} занят. Закройте старый сервер или укажите --port 8001.') from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Порт должен быть от 1 до 65535')
    model = HERE / 'tessdata' / 'rus.traineddata'
    if not model.is_file():
        raise RuntimeError('Не найдены русские веса. Сначала запустите setup_app.py.')
    verify_weights(model)
    os.environ['TESSDATA_PREFIX'] = str(model.parent)
    print(check_tesseract(model.parent), flush=True)
    try:
        import uvicorn
        from server import app
    except ImportError as exc:
        raise RuntimeError('Не установлены библиотеки. Запустите setup_app.py, затем Python из .venv-app.') from exc
    check_port(args.port)
    url = f'http://127.0.0.1:{args.port}'
    print(f'Приложение: {url}\nОстановка: Ctrl+C. Доступ только с этого компьютера.', flush=True)
    if not args.no_browser:
        timer = threading.Timer(1.5, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    uvicorn.run(app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('Ошибка запуска: ' + str(exc), file=sys.stderr)
        sys.exit(1)
