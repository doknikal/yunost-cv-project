"""Compare saved all-text runs, not OCR predictions against themselves as truth.

python docs/ocr_evaluation/compare_all_text.py all_text_corpus.json reliability_corpus.json
Paths may be absolute or relative to this script. Selected samples are development
excerpts, not an independent corpus-wide accuracy estimate.
"""
import argparse
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def normalize(text):
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=[а-я\u0451a-z])', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def region_text(page, box):
    x0, y0, x1, y1 = box
    lines = [line for block in page['blocks'] if block['is_text']
             for line in block['line_boxes']
             if x0 <= (line['x'] + line['right']) / 2 <= x1
             and y0 <= (line['y'] + line['bottom']) / 2 <= y1]
    return '\n'.join(line['text'] for line in sorted(lines, key=lambda ln: (ln['y'], ln['x'])))


def compare(before, after):
    old = {p['file']: p for p in before['pages']}
    new = {p['file']: p for p in after['pages']}
    if set(old) != set(new):
        raise ValueError('Page sets differ')
    geometry, changes = [], []
    for name in old:
        a, b = old[name], new[name]
        if 'error' in a or 'error' in b:
            continue
        if a['text_blocks'] != b['text_blocks'] or len(a['blocks']) != len(b['blocks']):
            geometry.append(name)
        for i, (ba, bb) in enumerate(zip(a['blocks'], b['blocks'])):
            fields = ['x', 'y', 'w', 'h', 'lines', 'is_text', 'source']
            if any(ba[k] != bb[k] for k in fields):
                geometry.append(name)
            for la, lb in zip(ba['line_boxes'], bb['line_boxes']):
                if any(la[k] != lb[k] for k in ['x', 'y', 'right', 'bottom']):
                    geometry.append(name)
                if la['text'] != lb['text']:
                    changes.append(dict(file=name, block=i, box=[la[k] for k in ['x', 'y', 'right', 'bottom']],
                                        before=la['text'], after=lb['text']))
    old_anchors = {(r['file'], r['region'], r['reference']): r for r in before['anchor_checks']}
    transitions = [dict(file=r['file'], reference=r['reference'], before=old_anchors[key]['present'],
                        after=r['present']) for r in after['anchor_checks']
                   for key in [(r['file'], r['region'], r['reference'])]
                   if key in old_anchors and old_anchors[key]['present'] != r['present']]
    samples = []
    for sample in json.loads((HERE / 'samples.json').read_text()):
        if sample['file'] not in new:
            continue
        reference = normalize(sample['text'])
        texts = [region_text(pages[sample['file']], sample['box']) for pages in [old, new]]
        errors = [distance(reference, normalize(t)) for t in texts]
        samples.append(dict(id=sample['id'], reference_chars=len(reference),
                            before_errors=errors[0], after_errors=errors[1],
                            before_cer=errors[0] / len(reference), after_cer=errors[1] / len(reference),
                            before=texts[0], after=texts[1]))
    return dict(pages=len(new), failures=[p['file'] for p in new.values() if 'error' in p],
                geometry_changes=sorted(set(geometry)), changed_lines=changes,
                changed_pages=len({c['file'] for c in changes}),
                refined_words=sum(b.get('refined_words', 0) for p in new.values() for b in p.get('blocks', [])),
                anchor_transitions=transitions, anchor_summary=after['anchor_summary'],
                selected_development_excerpts=samples,
                note='No full-page ground truth; geometry invariance and selected anchors are not corpus recall.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(json.loads((HERE / args.before).read_text()), json.loads((HERE / args.after).read_text()))
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(serialized)
    print(json.dumps({k: v for k, v in result.items() if k not in ['changed_lines', 'selected_development_excerpts']},
                     ensure_ascii=False, indent=2))
    for sample in result['selected_development_excerpts']:
        print(sample['id'], 'CER', round(sample['before_cer'], 4), '->', round(sample['after_cer'], 4))


if __name__ == '__main__':
    main()
