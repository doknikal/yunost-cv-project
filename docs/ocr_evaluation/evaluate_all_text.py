"""Reproduce genre-neutral extraction without overwriting prose-only snapshots.

TESSDATA_PREFIX="$PWD/.venv/tessdata/fast" .venv/bin/python \
    docs/ocr_evaluation/evaluate_all_text.py [--corpus]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yunost import TextExtractor

HERE = Path(__file__).resolve().parent


def normalize(text):
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=[а-яё])', '', text.lower())
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^\w]+', ' ', text).strip()


def anchor_checks(pages):
    """Use visual verse fixtures AND their prose/metadata anchors, not exclusions.

    These are selected development excerpts, not independent full-page recall.
    The older fixture's 'exclusion' terminology belongs to a genre-only task.
    """
    labels = json.loads((HERE / 'poetry_labels.json').read_text())
    checks = []
    by_name = {p['file']: p for p in pages}
    for page in labels['pages']:
        name = Path(page['image']).name
        if name not in by_name or 'error' in by_name[name]:
            continue
        text = normalize(by_name[name]['text'])
        for region in page['regions'] + page['prose_regions']:
            phrases = []
            for values in region.get('anchors', {}).values():
                phrases.extend(values)
            phrases.extend(region.get('verified_exclusion_anchors', []))
            if 'text' in region:
                phrases.append(region['text'])
            for phrase in phrases:
                reference = normalize(phrase)
                present = (reference in text if reference else
                           re.sub(r'\s+', '', phrase) in re.sub(r'\s+', '', by_name[name]['text']))
                checks.append(dict(file=name, region=region['id'], kind=region['kind'],
                                   reference=phrase, present=present))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', action='store_true', help='Process all 111 normalized pages')
    parser.add_argument('--rescore', action='store_true', help='Recheck saved anchors without running OCR')
    args = parser.parse_args()
    output = HERE / ('all_text_corpus.json' if args.corpus else 'all_text_examples.json')
    if args.rescore:
        result = json.loads(output.read_text())
        checks = anchor_checks(result['pages'])
        result['anchor_checks'] = checks
        result['anchor_summary'] = dict(present=sum(c['present'] for c in checks), total=len(checks))
        result['evaluation_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(result['anchor_summary']))
        return
    names = ['1967-12_p03.png', '1967-10_p03.png', '1957-12_p06.png',
             '1957-10_p05.png', '1957-12_p03.png', '1963-12_p07.png']
    paths = sorted((ROOT / 'data/norm').glob('*.png')) if args.corpus else [ROOT / 'data/norm' / n for n in names]
    overlays = ROOT / '.venv/evaluation/all_text_overlays'
    overlays.mkdir(parents=True, exist_ok=True)
    extractor = TextExtractor()
    pages = []
    start = time.monotonic()
    for path in paths:
        then = time.monotonic()
        data = path.read_bytes()
        row = dict(file=path.name, sha256=hashlib.sha256(data).hexdigest())
        try:
            result = extractor.extract(data)
            blocks = [b.to_dict() for b in result.blocks]
            for block in blocks:
                block.pop('words')  # Line boxes suffice for this smaller audit snapshot.
            row.update(text=result.text, text_blocks=len(result.text_blocks), blocks=blocks,
                       page_size=[result.gray.shape[1], result.gray.shape[0]])
            (overlays / path.name).write_bytes(result.overlay_png())
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        row['seconds'] = round(time.monotonic() - then, 3)
        pages.append(row)
        print(path.name, row.get('error', row.get('text_blocks')), row['seconds'], flush=True)
    checks = anchor_checks(pages)
    weights = Path(os.environ.get('TESSDATA_PREFIX', '')) / 'rus.traineddata'
    result = dict(mode='all_text', elapsed_seconds=round(time.monotonic() - start, 3),
                  tesseract=subprocess.run(['tesseract', '--version'], capture_output=True,
                                           text=True, check=True).stdout.splitlines()[0],
                  weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest() if weights.is_file() else None,
                  code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ['yunost/text.py', 'yunost/layout.py', 'yunost/ocr.py',
                                            'yunost/pipeline.py', 'server.py', 'static/index.html']},
                  fixture_sha256=hashlib.sha256((HERE / 'poetry_labels.json').read_bytes()).hexdigest(),
                  evaluation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note='Selected development anchors, not full-page ground truth or OCR recall. '
                       'Presence is punctuation/spacing/accent insensitive; other OCR errors still fail.',
                  anchor_summary=dict(present=sum(c['present'] for c in checks), total=len(checks)),
                  anchor_checks=checks, pages=pages)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(pages=len(pages), failures=sum('error' in p for p in pages),
                          anchors=result['anchor_summary']), ensure_ascii=False))


if __name__ == '__main__':
    main()
