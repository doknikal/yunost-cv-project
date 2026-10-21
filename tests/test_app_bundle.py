"""The standalone archive must contain working SQL seeds and offline chart assets."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import build_app


def test_archive_is_complete_reproducible_and_excludes_generated_database(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    for name in ['server.py', 'cvpipe.py', 'setup_app.py', 'run_app.py', 'import_corpus.py',
                 'requirements-app.txt', 'install_mac.command', 'start_mac.command',
                 'START_HERE.md', 'COURSEWORK.md', 'COURSEWORK_CHECK.md']:
        shutil.copyfile(root / name, tmp_path / name)
    for folder in ['yunost', 'static']:
        shutil.copytree(root / folder, tmp_path / folder, ignore=shutil.ignore_patterns('__pycache__'))
    (tmp_path / '1967-12').mkdir()
    (tmp_path / '1967-12/page_3.png').write_bytes(b'example fixture')
    (tmp_path / 'data').mkdir()
    for name in ['manifest.csv', 'features.csv']:
        shutil.copyfile(root / 'data' / name, tmp_path / 'data' / name)
    (tmp_path / 'data/corpus.sqlite3').write_bytes(b'local database, not for distribution')
    weights = tmp_path / 'tessdata'
    weights.mkdir()
    (weights / 'rus.traineddata').write_bytes(b'fixture weights')
    (weights / 'LICENSE').write_text('fixture license')
    monkeypatch.setattr(build_app, 'HERE', tmp_path)
    monkeypatch.setattr(build_app, 'ensure_weights', lambda: weights)
    monkeypatch.setattr(build_app, 'verify_weights', lambda _: None)
    archive_path = build_app.build()
    before = archive_path.read_bytes()
    build_app.build()
    assert archive_path.read_bytes() == before
    with zipfile.ZipFile(archive_path) as archive:
        files = {p.removeprefix('Yunost-OCR/'): archive.read(p) for p in archive.namelist()}
        manifest = json.loads(files.pop('MANIFEST.json'))
        assert manifest['version'] == '1.2.1'
        assert manifest['files'] == {p: hashlib.sha256(content).hexdigest() for p, content in files.items()}
        required = {'import_corpus.py', 'yunost/storage.py', 'yunost/analytics.py',
                    'static/analytics.html', 'static/analytics.css', 'static/analytics.js',
                    'yunost/charts.py', 'static/vendor/README.md',
                    'data/manifest.csv', 'data/features.csv', 'COURSEWORK.md', 'COURSEWORK_CHECK.md'}
        assert required.issubset(files)
        assert not any(p.endswith('.sqlite3') or '__pycache__' in p for p in files)
        for info in archive.infolist():
            assert ((info.external_attr >> 16) & 0o777) == (0o755 if info.filename.endswith('.command') else 0o644)
