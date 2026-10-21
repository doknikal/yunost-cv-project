"""Build a small app archive, not the 2GB research repository or a copied venv."""
import hashlib
import json
from pathlib import Path
import zipfile

from setup_app import HERE, ensure_weights, verify_weights, WEIGHTS_SHA256

VERSION = '1.2.1'


def build():
    model_folder = ensure_weights()
    verify_weights(model_folder / 'rus.traineddata')
    paths = [HERE / name for name in ['server.py', 'cvpipe.py', 'setup_app.py', 'run_app.py',
             'requirements-app.txt', 'install_mac.command', 'start_mac.command', 'START_HERE.md',
             'import_corpus.py', 'COURSEWORK.md', 'COURSEWORK_CHECK.md']]
    paths += sorted((HERE / 'yunost').glob('*.py'))
    paths += sorted(path for path in (HERE / 'static').rglob('*') if path.is_file()
                    and path.suffix in {'.html', '.css', '.js', '.md'})
    paths += [HERE / 'data/manifest.csv', HERE / 'data/features.csv',
              model_folder / 'rus.traineddata', model_folder / 'LICENSE']
    entries = {str(p.relative_to(HERE)): p.read_bytes() for p in paths}
    entries['examples/page_3.png'] = (HERE / '1967-12/page_3.png').read_bytes()
    report = HERE / 'docs/ocr_evaluation/RELIABILITY.md'
    if report.is_file():
        entries['RELIABILITY.md'] = report.read_bytes()
    manifest = dict(version=VERSION, weights_sha256=WEIGHTS_SHA256,
                    files={name: hashlib.sha256(data).hexdigest() for name, data in sorted(entries.items())})
    entries['MANIFEST.json'] = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    destination = HERE / 'dist' / f'Yunost-OCR-{VERSION}.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo('Yunost-OCR/' + name, date_time=(2026, 10, 21, 0, 0, 0))
            info.create_system = 3
            mode = 0o755 if name.endswith('.command') else 0o644
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_text(f'{digest}  {destination.name}\n')
    print(f'{destination}\n{destination.stat().st_size / 1024 / 1024:.1f} MiB; SHA256 {digest}')
    return destination


if __name__ == '__main__':
    build()
