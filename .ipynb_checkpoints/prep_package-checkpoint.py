"""Регенерация нормализованных страниц из исходных сканов.
Запуск:  python prep_package.py
Ожидает, что папка с исходниками '1967-12' лежит рядом со скриптом (./1967-12) либо
задана переменной окружения YUNOST_SRC."""
import os, re, glob, csv, sys
sys.stdout.reconfigure(encoding="utf-8")   # печать кириллицы в консоль Windows (cp1252)
from PIL import Image
import pillow_heif; pillow_heif.register_heif_opener()
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("YUNOST_SRC", os.path.join(HERE, "1967-12"))
NORM_DIR = os.path.join(HERE, "data", "norm"); os.makedirs(NORM_DIR, exist_ok=True); TARGET_H = 2000

def issue_from_dir(d):
    m = re.match(r'(\d{4})-(\d{1,2})$', os.path.basename(d))
    return (int(m.group(1)), int(m.group(2))) if m else None
def page_key(f):
    m = re.search(r'page_(\d+)', f);  n = re.search(r'IMG_(\d+)', f)
    return (0,int(m.group(1))) if m else (1,int(n.group(1))) if n else (2,f)

dirs = []
top = issue_from_dir(SRC)
if top: dirs.append((SRC, top))
for d in glob.glob(os.path.join(SRC, "1957", "*")):
    if os.path.isdir(d) and issue_from_dir(d): dirs.append((d, issue_from_dir(d)))

rows = []
for dp,(y,mo) in dirs:
    fs = sorted([f for f in os.listdir(dp) if f.lower().endswith(('png','jpg','jpeg','heic'))], key=page_key)
    for pi,f in enumerate(fs,1):
        src_fmt = 'photo' if f.lower().endswith('heic') else 'scan'
        im = Image.open(os.path.join(dp,f)).convert('RGB'); ow,oh = im.size
        nw = int(round(ow*TARGET_H/oh)); out = f"{y}-{mo:02d}_p{pi:02d}.png"
        im.resize((nw,TARGET_H), Image.LANCZOS).save(os.path.join(NORM_DIR,out))
        rows.append(dict(issue=f"{y}-{mo:02d}",year=y,month=mo,page=pi,source=src_fmt,
                         orig_w=ow,orig_h=oh,norm_w=nw,norm_h=TARGET_H,file=out,
                         orig_path=os.path.join(dp,f)))
rows.sort(key=lambda r:(r['year'],r['month'],r['page']))
with open(os.path.join(HERE,"data","manifest.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("OK:",len(rows),"страниц,",len(set(r['issue'] for r in rows)),"номеров;",dict(Counter(r['source'] for r in rows)))
