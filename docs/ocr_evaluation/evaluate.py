"""Reproduce current OCR/preprocessing and page-presence/region diagnostics.

TESSDATA_PREFIX="$PWD/.venv/tessdata/fast" .venv/bin/python \
  docs/ocr_evaluation/evaluate.py [--corpus]

Does not overwrite historical engine outputs. Re-scores their saved predictions
against the current, typo-corrected references in corrected_metrics.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark import distance, normalize
import cvpipe
from yunost import ProseExtractor
from yunost.ocr import TesseractEngine
from yunost.preprocessing import prepare_text

HERE = Path(__file__).resolve().parent


def score_samples(predictions, references):
    rows = []
    for row in predictions:
        ref = normalize(references[row['id']]['text'])
        errors = distance(ref, normalize(row['text']))
        rows.append(dict(id=row['id'], reference_chars=len(ref), char_errors=errors,
                         cer=errors / len(ref)))
    return dict(samples=rows, weighted_cer=sum(r['char_errors'] for r in rows) /
                sum(r['reference_chars'] for r in rows))


def page_metrics(pages, labels):
    by_name = {p['file']: p for p in pages}
    rows = []
    for label in labels['pages']:
        page = by_name[label['file']]
        blocks = [b for b in page['blocks'] if b['is_prose']]
        width, height = label['image_size']
        mask = np.zeros((height, width), np.uint8)
        for b in blocks:
            mask[b['y']:b['y'] + b['h'], b['x']:b['x'] + b['w']] = 1
        regions = []
        for region in label['regions']:
            x0, y0, x1, y1 = region['bounds']
            regions.append(dict(kind=region['kind'], bounds=region['bounds'],
                                box_coverage=float(mask[y0:y1, x0:x1].mean())))
        rows.append(dict(file=label['file'], expected=label['label'],
                         predicted_prose=bool(blocks),
                         correct_presence=bool(blocks) == (label['label'] == 'hasprose'),
                         regions=regions))
    return dict(correct_presence=sum(r['correct_presence'] for r in rows),
                total=len(rows),
                false_positive_pages=[r['file'] for r in rows if
                                      r['predicted_prose'] and r['expected'] == 'pureverse'],
                false_negative_pages=[r['file'] for r in rows if
                                      not r['predicted_prose'] and r['expected'] == 'hasprose'],
                pages=rows)


def substring_errors(reference, text):
    """Levenshtein distance to the best matching substring, not to the whole page."""
    previous = [0] * (len(text) + 1)
    for i, char in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(text, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char != other)))
        previous = current
    return min(previous)


def completeness_metrics(pages, labels):
    by_name = {p['file']: p for p in pages}
    rows = []
    for page in labels['pages']:
        if page['file'] not in by_name:
            continue
        prediction = by_name[page['file']]
        width, height = page['size']
        mask = np.zeros((height, width), np.uint8)
        for block in prediction['blocks']:
            if block['is_prose']:
                mask[block['y']:block['y'] + block['h'], block['x']:block['x'] + block['w']] = 1
        for passage in page['missing_passages']:
            x0, y0, x1, y1 = passage['bounds']
            ref = normalize(passage.get('context_reference', passage['reference'])).lower()
            errors = substring_errors(ref, normalize(prediction['text']).lower())
            rows.append(dict(file=page['file'], id=passage['id'], stage=passage['stage'],
                             box_coverage=float(mask[y0:y1, x0:x1].mean()),
                             reference_chars=len(ref), best_substring_errors=errors,
                             best_substring_cer=errors / len(ref)))
    return dict(note='Selected previously missing excerpts, not full-page OCR recall. '
                     'Bounding-box coverage is not glyph coverage. Lowercased substring CER '
                     'may find a different occurrence; inspect source and output together.',
                passages=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', action='store_true')
    parser.add_argument('--output-prefix', default='completeness',
                        help='Output basename (default: completeness; preserves the earlier snapshot)')
    args = parser.parse_args()
    samples_bytes = (HERE / 'samples.json').read_bytes()
    references = {s['id']: s for s in json.loads(samples_bytes)}
    reference_hash = hashlib.sha256(samples_bytes).hexdigest()
    code_hashes = {name: hashlib.sha256((ROOT / 'yunost' / name).read_bytes()).hexdigest()
                   for name in ['blocks.py', 'ocr.py', 'pipeline.py', 'preprocessing.py', 'recovery.py']}
    corrected = dict(reference_sha256=reference_hash,
                     note='Re-scored saved predictions; reference typo Татарской→Тагарской. '
                          'Historical result JSON files remain unchanged.', engines={})
    for name in ['tesseract-fast', 'tesseract-best', 'easyocr', 'paddle-eslav', 'paddle-cyrillic']:
        saved = json.loads((HERE / (name + '.json')).read_text())
        corrected['engines'][name] = score_samples(saved['samples'], references)
    (HERE / 'corrected_metrics.json').write_text(json.dumps(corrected, ensure_ascii=False, indent=2) + '\n')
    engine = TesseractEngine()
    predictions = []
    for sample in references.values():
        flat = cvpipe.normalize_illumination(cv2.imread(str(ROOT / 'data/norm' / sample['file']), 0))
        x0, y0, x1, y1 = sample['box']
        crop, angle, noise = prepare_text(flat[y0:y1, x0:x1])
        text = '\n'.join(line.text for line in engine.read_lines(crop))
        predictions.append(dict(id=sample['id'], text=text, angle=angle, noise=noise))
    labels = json.loads((HERE / 'page_labels.json').read_text())
    filenames = sorted(p.name for p in (ROOT / 'data/norm').glob('*.png')) if args.corpus else [
        p['file'] for p in labels['pages']]
    overlays = ROOT / '.venv/evaluation' / (args.output_prefix + '_overlays')
    overlays.mkdir(parents=True, exist_ok=True)
    labeled_names = {p['file'] for p in labels['pages']}
    extractor = ProseExtractor(engine=engine)
    pages = []
    for filename in filenames:
        started = time.perf_counter()
        page = extractor.extract((ROOT / 'data/norm' / filename).read_bytes())
        pages.append(dict(file=filename, seconds=time.perf_counter() - started,
                          prose_blocks=len(page.prose), text=page.text,
                          blocks=[b.to_dict() for b in page.blocks]))
        if filename in labeled_names or page.prose:
            (overlays / filename).write_bytes(page.overlay_png())
        print(filename, len(page.prose), round(pages[-1]['seconds'], 2), flush=True)
    result = dict(engine='tesseract-fast', reference_sha256=reference_hash, code_sha256=code_hashes,
                  samples=predictions, sample_metrics=score_samples(predictions, references),
                  pages=pages, metrics=page_metrics(pages, labels))
    baseline = json.loads((HERE / 'tesseract-fast_corpus.json').read_text())
    result['baseline_metrics'] = page_metrics(baseline['pages'], labels)
    completeness_path = HERE / 'completeness_labels.json'
    if completeness_path.exists():
        completeness = json.loads(completeness_path.read_text())
        result['completeness_labels_sha256'] = hashlib.sha256(completeness_path.read_bytes()).hexdigest()
        result['completeness'] = completeness_metrics(pages, completeness)
        previous = json.loads((HERE / 'improved_corpus.json').read_text())
        result['previous_completeness'] = completeness_metrics(previous['pages'], completeness)
    suffix = '_corpus' if args.corpus else ''
    (HERE / (args.output_prefix + suffix + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result['metrics'].items() if k != 'pages'}, indent=2))
    print('Crop weighted CER:', result['sample_metrics']['weighted_cer'])


if __name__ == '__main__':
    main()
