"""Small reproducible OCR comparison; does NOT change the production engine.

Run from repository root:
  .venv/bin/python docs/ocr_evaluation/benchmark.py --engine tesseract-fast
  .venv/bin/python docs/ocr_evaluation/benchmark.py --engine tesseract-best
  .venv/bin/python docs/ocr_evaluation/benchmark.py --engine paddle-eslav
  .venv/bin/python docs/ocr_evaluation/benchmark.py --engine paddle-cyrillic
  .venv/bin/python docs/ocr_evaluation/benchmark.py --engine easyocr

Optional dependencies: rapidocr==3.10.0 onnxruntime==1.19.2.
EasyOCR: easyocr==1.7.2, torch==2.8.0; CPU only, 4 Torch threads.
EasyOCR weights are downloaded to .venv/evaluation/easyocr on first use.
Put rus.traineddata in .venv/tessdata/{fast,best}/ (upstream tessdata repos).
Samples are manually transcribed excerpts, NOT a representative held-out corpus.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import cvpipe
from yunost.ocr import OcrEngine, OcrLine, TesseractEngine
from yunost.pipeline import ProseExtractor

HERE = Path(__file__).resolve().parent


def fragment_lines(fragments):
    """Reassemble detected fragments into physical lines, preserving indents."""
    rows = []
    for fragment in sorted(fragments):
        cy, x, h, text = fragment
        if rows and abs(cy - np.mean([f[0] for f in rows[-1]])) < 0.5 * h:
            rows[-1].append(fragment)
        else:
            rows.append([fragment])
    return [OcrLine(" ".join(f[3] for f in sorted(row, key=lambda f: f[1])),
                    int(min(f[1] for f in row))) for row in rows]


class EasyOcrEngine(OcrEngine):
    """Evaluation adapter: CRAFT + Cyrillic recognition, CPU, default decoding."""

    name = "easyocr"

    def __init__(self):
        import easyocr
        import torch

        torch.set_num_threads(4)
        if torch.get_num_interop_threads() > 4:
            torch.set_num_interop_threads(1)
        weights = ROOT / ".venv/evaluation/easyocr"
        self.model = easyocr.Reader(
            ["ru", "en"], gpu=False,
            model_storage_directory=str(weights),
            user_network_directory=str(weights / "user_network"),
            verbose=False,
        )

    def read_lines(self, image):
        fragments = []
        for box, text, _confidence in self.model.readtext(
                image, detail=1, paragraph=False, workers=0):
            box = np.asarray(box)
            x = float(np.min(box[:, 0]))
            cy = float(np.mean(box[:, 1]))
            h = float(np.max(box[:, 1]) - np.min(box[:, 1]))
            fragments.append((cy, x, h, text))
        return fragment_lines(fragments)


class PaddleEngine(OcrEngine):
    """Evaluation adapter: PP-OCRv5, CPU, 4 threads, per-block line detection."""

    def __init__(self, lang):
        from rapidocr import RapidOCR, LangRec, OCRVersion, ModelType
        self.model = RapidOCR(params={
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.model_type": ModelType.MOBILE,
            "Det.limit_side_len": 736,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.lang_type": LangRec.ESLAV if lang == "eslav" else LangRec.CYRILLIC,
            "Global.use_cls": False,
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        })

    def read_lines(self, image):
        result = self.model(image)
        if result.boxes is None or result.txts is None:
            return []
        # Detection can split a justified line into several pieces. Reassemble
        # before applying the existing capital-start and paragraph heuristics.
        fragments = []
        for box, text in zip(result.boxes, result.txts):
            x = float(np.min(box[:, 0]))
            cy = float(np.mean(box[:, 1]))
            h = float(np.max(box[:, 1]) - np.min(box[:, 1]))
            fragments.append((cy, x, h, text))
        return fragment_lines(fragments)


def normalize(text):
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=[а-яёa-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=[
        "tesseract-fast", "tesseract-best", "paddle-eslav", "paddle-cyrillic",
        "easyocr"])
    parser.add_argument("--corpus", action="store_true", help="also run all 111 pages")
    args = parser.parse_args()
    if args.engine.startswith("tesseract"):
        os.environ["TESSDATA_PREFIX"] = str(ROOT / ".venv/tessdata" / args.engine.split("-")[1])
        engine = TesseractEngine()
    elif args.engine == "easyocr":
        engine = EasyOcrEngine()
    else:
        engine = PaddleEngine(args.engine.split("-")[1])
    # Warm up separately from timing.
    engine.read_lines(np.full((60, 300), 255, dtype=np.uint8))
    samples = json.loads((HERE / "samples.json").read_text())
    result = {"engine": args.engine, "samples": [], "pages": []}
    for s in samples:
        gray = cv2.imread(str(ROOT / "data/norm" / s["file"]), 0)
        # Match production: normalize the full page, then crop (not vice versa).
        flat = cvpipe.normalize_illumination(gray)
        x0, y0, x1, y1 = s["box"]
        start = time.perf_counter()
        predicted = "\n".join(ln.text for ln in engine.read_lines(flat[y0:y1, x0:x1]))
        seconds = time.perf_counter() - start
        ref, hyp = normalize(s["text"]), normalize(predicted)
        errors = distance(ref, hyp)
        row = dict(id=s["id"], kind=s["kind"], text=predicted, seconds=seconds,
                   reference_chars=len(ref), char_errors=errors, cer=errors / len(ref))
        result["samples"].append(row)
        print(s["id"], "CER", round(row["cer"], 4), "seconds", round(seconds, 3), flush=True)
    pages = sorted((ROOT / "data/norm").glob("*.png")) if args.corpus else [
        ROOT / "data/norm" / f for f in [
            "1957-10_p05.png", "1957-07_p02.png", "1957-11_p02.png",
            "1963-12_p07.png", "1965-01_p04.png", "1965-02_p01.png",
            "1965-02_p09.png", "1967-10_p03.png"]]
    extractor = ProseExtractor(engine=engine)
    for p in pages:
        start = time.perf_counter()
        page = extractor.extract(p.read_bytes())
        row = dict(file=p.name, seconds=time.perf_counter()-start,
                   prose_blocks=len(page.prose), text=page.text,
                   blocks=[b.to_dict() for b in page.blocks])
        result["pages"].append(row)
        print(p.name, row["prose_blocks"], round(row["seconds"], 3), flush=True)
    suffix = "_corpus" if args.corpus else ""
    (HERE / f"{args.engine}{suffix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
