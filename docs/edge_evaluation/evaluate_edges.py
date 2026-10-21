"""Evaluate physical edges, never genre; write new evidence, never old CSVs.

Example: .venv/bin/python docs/edge_evaluation/evaluate_edges.py \
    --split development --output .venv/evaluation/edges/development.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cvpipe


def annotation(path):
    data = json.loads(path.read_text())
    if isinstance(data.get('source'), dict):
        image = data['source']['path']
        digest = data['source']['sha256']
        bbox = data['region']['crop_bbox_xyxy']
        lines = [dict(top=r['y_top_px'], bottom=r['y_bottom_px'], right=r['right_x_px'])
                 for r in data['lines']]
        uncertainty = data['coordinate_conventions']['default_uncertainty_px']['right_x_px']
        split = data.get('split', 'development')
    elif isinstance(data.get('source'), str):
        image, digest, bbox = data['source'], data['source_sha256'], data['bbox']
        lines = [dict(top=r['y_top'], bottom=r['y_bottom'], right=r['x_right'])
                 for r in data['lines']]
        uncertainty = data['uncertainty_px']['x_right']
        split = data['split']
    else:
        image, digest, bbox = data['image'], data['sha256'], data['bbox']
        lines, uncertainty, split = data['lines'], data['uncertainty_px'], data['split']
    if hashlib.sha256((ROOT / image).read_bytes()).hexdigest() != digest:
        raise ValueError('Source image differs from the reviewed annotation: ' + image)
    return dict(image=image, bbox=bbox, lines=lines, uncertainty=uncertainty, split=split)


def legacy_lines(mask, bbox):
    """Exact physical-band logic of the frozen historical scalar, with diagnostics."""
    x0, y0, x1, y1 = bbox
    region = mask[y0:y1, x0:x1]
    profile = region.sum(axis=1) / 255.0
    if not profile.size or profile.max() <= 0:
        return []
    rows = np.flatnonzero(profile > .2 * profile.max())
    if len(rows) < 4:
        return []
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 3) + 1)
    lines = []
    for group in groups:
        if not len(group):
            continue
        top, bottom = int(group[0]), int(group[-1]) + 1
        columns = np.flatnonzero(region[top:bottom].any(axis=0))
        if len(columns):
            lines.append(dict(right=float(x0 + columns[-1] + 1), center_y=y0 + (top + bottom) / 2,
                              box=[x0 + int(columns[0]), y0 + top, x0 + int(columns[-1]) + 1, y0 + bottom]))
    return lines


def score(predicted, expected, uncertainty):
    # Match by vertical position ONLY; endpoint agreement must not select matches.
    costs = np.array([[abs(p['center_y'] - (e['top'] + e['bottom']) / 2)
                       for e in expected] for p in predicted])
    matches = []
    if predicted and expected:
        rows, columns = linear_sum_assignment(costs)
        for i, j in zip(rows.tolist(), columns.tolist()):
            target = expected[j]
            tolerance = .65 * (target['bottom'] - target['top']) + uncertainty
            if costs[i, j] <= tolerance:
                matches.append((i, j))
    errors = [abs(predicted[i]['right'] - expected[j]['right']) for i, j in matches]
    return dict(expected_lines=len(expected), predicted_lines=len(predicted), matched_lines=len(matches),
                precision=len(matches) / len(predicted) if predicted else 0,
                recall=len(matches) / len(expected) if expected else 0,
                endpoint_mae_px=float(np.mean(errors)) if errors else None,
                endpoint_max_error_px=max(errors) if errors else None,
                within_annotation_tolerance=sum(error <= uncertainty for error in errors),
                endpoint_errors_px=errors, matches=matches,
                unmatched_expected=[i for i in range(len(expected)) if i not in {j for _, j in matches}],
                unmatched_predicted=[i for i in range(len(predicted)) if i not in {j for j, _ in matches}])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['development', 'held_out', 'all'], default='development')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new run path. Historical evidence is not overwritten.')
    entries = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(__file__).parent.glob('manual_*.json')):
        label = annotation(path)
        if args.split != 'all' and label['split'] != args.split:
            continue
        gray = cvpipe.load_gray(str(ROOT / label['image']))
        mask = cvpipe.ink_mask(gray)
        x0, y0, x1, y1 = label['bbox']
        result = cvpipe.analyze_right_edge(mask, (x0, x1), y0, y1)
        old_lines = legacy_lines(mask, label['bbox'])
        legacy_value = cvpipe.right_edge_raggedness(mask, (x0, x1), y0, y1, method='legacy')
        reference = float(np.std([line['right'] for line in label['lines']]) / (x1 - x0))
        entry = dict(annotation=path.name, annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     image=label['image'], split=label['split'], uncertainty_px=label['uncertainty'],
                     normalization_width=x1-x0, manual_raw_endpoint_sd=reference,
                     legacy=dict(score(old_lines, label['lines'], label['uncertainty']),
                                 value=float(legacy_value) if np.isfinite(legacy_value) else None,
                                 lines=old_lines),
                     improved=dict(score(result['lines'], label['lines'], label['uncertainty']),
                                   value=result['value'], correction_angle=result['correction_angle'],
                                   raw_endpoint_sd=float(np.std([line['right'] for line in result['lines']])
                                                         / (x1 - x0)) if result['line_count'] >= 4 else None,
                                   warnings=result['warnings'], lines=result['lines']))
        # A review image is evidence, not the source of the annotations.
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for target in label['lines']:
            y = round((target['top'] + target['bottom']) / 2)
            cv2.line(vis, (target['right']-7, y), (target['right']+7, y), (180, 0, 180), 2)
        for line in result['lines']:
            x, y, r, b = line['box']
            cv2.rectangle(vis, (x, y), (r, b), (0, 150, 0), 1)
        overlay = args.output.parent / (args.output.stem + '_' + path.stem + '.png')
        if overlay.exists():
            raise FileExistsError(overlay)
        cv2.imwrite(str(overlay), vis[y0:y1, x0:x1])
        entries.append(entry)
    report = dict(method='components-v2', split=args.split, regions=entries,
                  note='Model-visual approximate annotations, not independent human gold. '
                       'Selected single-column crops, not end-to-end page recall or genre accuracy. '
                       'Manual SD is in raw image coordinates; deskewed SD is not the same target on tilted photos.',
                  code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ['cvpipe.py', 'yunost/edges.py', 'yunost/preprocessing.py',
                                            'docs/edge_evaluation/evaluate_edges.py']},
                  versions=dict(opencv=cv2.__version__, numpy=np.__version__))
    with args.output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    for entry in entries:
        print(entry['annotation'], {name: {k: v for k, v in entry[name].items()
                                          if k in {'matched_lines', 'predicted_lines', 'endpoint_mae_px', 'value'}}
                                   for name in ['legacy', 'improved']})


if __name__ == '__main__':
    main()
