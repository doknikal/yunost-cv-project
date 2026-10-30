"""Safe standalone-app installer; never deletes/recreates an existing environment."""
import hashlib
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import urllib.request
import venv

HERE = Path(__file__).resolve().parent
WEIGHTS_SHA256 = 'e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225'
MODEL_REVISION = '87416418657359cb625c412a48b6e1d6d41c29bd'
MODEL_BASE = f'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{MODEL_REVISION}'


def verify_weights(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError(f'Неверная контрольная сумма весов: {path}. Файл не изменен.')


def ensure_weights():
    folder = HERE / 'tessdata'
    folder.mkdir(exist_ok=True)
    path = folder / 'rus.traineddata'
    if path.exists():
        verify_weights(path)
    else:
        print('Загружаем бесплатные русские веса Tesseract…', flush=True)
        temporary = None
        try:
            with urllib.request.urlopen(MODEL_BASE + '/rus.traineddata', timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=folder, delete=False) as output:
                    temporary = Path(output.name)
                    shutil.copyfileobj(response, output)
            verify_weights(temporary)
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    license_path = folder / 'LICENSE'
    if not license_path.exists():
        with urllib.request.urlopen(MODEL_BASE + '/LICENSE', timeout=30) as response:
            license_path.write_bytes(response.read())
    return folder


def check_tesseract(tessdata=None):
    executable = shutil.which('tesseract')
    if not executable:
        raise RuntimeError('Нужен Tesseract 5: macOS — brew install tesseract; '
                           'Ubuntu/Debian — sudo apt install tesseract-ocr. '
                           'Windows — установщик UB Mannheim, затем добавьте Tesseract в PATH.')
    result = subprocess.run([executable, '--version'], capture_output=True, text=True,
                            timeout=10, check=True)
    version = result.stdout.splitlines()[0]
    if int(version.split()[1].split('.')[0]) < 5:
        raise RuntimeError('Обновите Tesseract до версии 5 или новее')
    if tessdata is not None:
        env = dict(os.environ, TESSDATA_PREFIX=str(tessdata))
        langs = subprocess.run([executable, '--list-langs'], env=env, capture_output=True,
                               text=True, timeout=10, check=True).stdout.splitlines()[1:]
        if 'rus' not in langs:
            raise RuntimeError('Tesseract не видит русские веса')
    return version


def main():
    if not (3, 9) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError('Для проверенного набора библиотек нужен Python 3.9–3.12. '
                           'На macOS: brew install python@3.12; '
                           '/opt/homebrew/bin/python3.12 setup_app.py')
    print(check_tesseract(), flush=True)
    model = ensure_weights()
    print(check_tesseract(model), '— русский язык доступен', flush=True)
    environment = HERE / '.venv-app'
    marker = environment / '.yunost-app'
    if environment.exists():
        if not marker.is_file():
            raise RuntimeError(f'{environment} уже существует и не создан этим установщиком. '
                               'Ничего не удалено. Распакуйте приложение в отдельную папку.')
    else:
        venv.EnvBuilder(with_pip=True).create(environment)
        marker.write_text('Yunost OCR application environment\n')
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        raise RuntimeError('Окружение повреждено. Распакуйте приложение в новую папку; старое не изменено.')
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(HERE / 'requirements-app.txt')], check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    command = [str(python), str(HERE / 'run_app.py')]
    display = subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command)
    print('Готово. Запуск: ' + display, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('Ошибка установки: ' + str(exc), file=sys.stderr)
        sys.exit(1)
