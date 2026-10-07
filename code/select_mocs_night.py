"""Select the MOCS night-time images and split them into the translation pool and the night-time validation set.

Input: the output folder of scan_mocs_night.py and the inspection decisions (contact-sheet ranks).
--accept lists the ranks judged to be night-time photographs lit only by artificial light; --reject removes ranks inside
accepted ranges (indoor, underground, twilight and under-exposed daytime images). Near-identical frames of the same scene
are reduced to one image (correlation of downsampled grey-scale thumbnails >= 0.95, the darkest image is kept). The
remaining images are split at random (seed 2026) into 70% for the translation pool and 30% for the validation set.

The decisions used in the study (ranks 0-559 inspected):
  --accept "0-319,320,322,325,326,327,329,330,331,332,335,336,337,338,343,344,345,348,349,353,375,387,388,389,390,396,
            398,399,403,404,405,407,408,409,410,411,412,413,418,419,423,428,437,446,450,451,454,456,459,460,461,464,469,
            479,488,496,497,505,510,513,518,521,531,544,545,546,550"
  --reject "100,102,105,162,211,213,214,230,232,233,257,262,263,264,287,289,297,300,305,310,311,312,313,314,316"
which give 361 night-time images, 293 after near-duplicate removal, 205 for the translation pool and 88 for validation
(171 annotated workers).

Usage: python select_mocs_night.py --scan <scan folder> --accept "<ranks>" --reject "<ranks>" --out <folder>
Output: night_split.json (pool file names; validation file names with image size and worker boxes) and summary.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

CELL, COLS, PER_SHEET = 180, 10, 80


def ranks(spec):
    out = set()
    for part in filter(None, (p.strip() for p in (spec or '').split(','))):
        a, _, b = part.partition('-')
        out.update(range(int(a), int(b or a) + 1))
    return out


def thumbnail_feature(scan, r):
    """24 x 18 z-scored grey thumbnail of contact-sheet rank r."""
    sh = Image.open(Path(scan, 'sheets', f'sheet_{r // PER_SHEET:03d}.jpg')).convert('L')
    k = r % PER_SHEET
    x, y = (k % COLS) * CELL + 2, (k // COLS) * (CELL + 14) + 2
    c = np.asarray(sh.crop((x, y, x + CELL - 4, y + CELL - 4)), dtype=np.float32)
    nw = c < 235   # trim the white cell margin (rows and columns that are mostly non-white)
    rows, cols = np.where(nw.mean(1) > 0.5)[0], np.where(nw.mean(0) > 0.5)[0]
    if len(rows) > 10 and len(cols) > 10:
        c = c[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    v = np.asarray(Image.fromarray(c.astype(np.uint8)).resize((24, 18), Image.BILINEAR), dtype=np.float32).ravel()
    v = (v - v.mean()) / (v.std() + 1e-6)
    return v / np.sqrt(len(v))


def dedup(scan, ranks_sorted, thr):
    """Keep the darkest image of each group of near-identical images (thumbnail correlation >= thr)."""
    feats = {r: thumbnail_feature(scan, r) for r in ranks_sorted}
    kept, dup = [], {}
    for r in ranks_sorted:
        best = max(((float(feats[r] @ feats[k]), k) for k in kept), default=(-1.0, None))
        if best[0] >= thr:
            dup[r] = best
        else:
            kept.append(r)
    return kept, dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', required=True)
    ap.add_argument('--accept', required=True)
    ap.add_argument('--reject', default='')
    ap.add_argument('--dedup', type=float, default=0.95)
    ap.add_argument('--pool_share', type=float, default=0.7)
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    info = json.loads(Path(a.scan, 'mocs_info.json').read_text())
    ranked = json.loads(Path(a.scan, 'ranked_darkest.json').read_text())
    keep = sorted(ranks(a.accept) - ranks(a.reject))
    keep, dup = dedup(a.scan, keep, a.dedup)
    by_rank = {r['rank']: r['file'] for r in ranked}
    night = sorted(by_rank[r] for r in keep)
    random.Random(2026).shuffle(night)
    cut = round(a.pool_share * len(night))
    pool, val = sorted(night[:cut]), sorted(night[cut:])
    split = {'pool': pool, 'val': {f: {'wh': [info[f]['w'], info[f]['h']], 'workers': info[f]['workers']} for f in val}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, 'night_split.json').write_text(json.dumps(split))
    summary = dict(accepted=len(keep) + len(dup), near_duplicates_removed=len(dup), night_images=len(night),
                   pool=len(pool), val=len(val), val_worker_boxes=sum(len(v['workers']) for v in split['val'].values()))
    Path(a.out, 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
