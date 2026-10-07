"""Rank the annotated MOCS images by luminance and draw numbered contact sheets of the darkest images.

MOCS (An et al., 2021) provides COCO-style annotation files (instances_<split>.json) and the image folders of its public
release. For every annotated image, the script records the image size, the worker boxes (categories whose name contains
'worker') and the mean luminance of the whole image and of its upper third (0-255, on a 320-pixel thumbnail). The 2,400
images with the darkest upper third are drawn on numbered contact sheets (80 images per sheet), which were inspected to
select night-time photographs lit only by artificial light (file_lists/mocs_night_candidates.csv).

Usage: python scan_mocs_night.py <MOCS root folder> <output folder>
Output: mocs_info.json, ranked_darkest.json and sheets/sheet_NNN.jpg in the output folder
"""
import collections
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

N_SHEET, COLS, ROWS, CELL = 2400, 10, 8, 180


def index_images(root):
    index = collections.defaultdict(list)   # file name -> paths (names may repeat across MOCS splits)
    for p in glob.glob(f'{root}/**/*.jpg', recursive=True):
        index[os.path.basename(p)].append(p)
    return index


def resolve(index, key):
    """key = '<split>/<file_name>' -> image path inside that split's folder."""
    split, fn = key.split('/', 1)
    c = index.get(os.path.basename(fn), [])
    inside = [q for q in c if split in q.replace(os.sep, '/')]
    return inside[0] if inside else (c[0] if len(c) == 1 else None)


def luminance(path):
    """Mean luminance of the whole image and of its upper third (0-255), on a 320-pixel thumbnail."""
    im = Image.open(path).convert('L')
    im.thumbnail((320, 320))
    a = np.asarray(im, dtype=np.float32)
    return float(a.mean()), float(a[: max(1, a.shape[0] // 3)].mean())


def main(root, out):
    os.makedirs(f'{out}/sheets', exist_ok=True)
    index = index_images(root)
    info = {}
    for ann in sorted(glob.glob(f'{root}/**/*instances*.json', recursive=True)):
        split = os.path.basename(ann).replace('instances_', '').replace('.json', '')
        d = json.load(open(ann))
        workers = {c['id'] for c in d['categories'] if 'worker' in c['name'].lower()}
        by_id = {}
        for im in d['images']:
            key = f"{split}/{im['file_name']}"
            info[key] = dict(split=split, id=im['id'], w=im['width'], h=im['height'], workers=[], n_obj=0)
            by_id[im['id']] = key
        for an in d['annotations']:
            key = by_id.get(an['image_id'])
            if key is not None:
                info[key]['n_obj'] += 1
                if an['category_id'] in workers:
                    info[key]['workers'].append([round(v, 1) for v in an['bbox']])
    files = [k for k in info if resolve(index, k) is not None]
    with ProcessPoolExecutor() as ex:
        lums = list(ex.map(luminance, [resolve(index, k) for k in files], chunksize=64))
    for k, (m, u) in zip(files, lums):
        info[k]['lum'], info[k]['lum_upper'] = m, u
    json.dump(info, open(f'{out}/mocs_info.json', 'w'))
    ranked = sorted(files, key=lambda k: info[k]['lum_upper'])[:N_SHEET]
    for s in range(0, len(ranked), COLS * ROWS):
        sheet = Image.new('RGB', (COLS * CELL, ROWS * (CELL + 14)), 'white')
        draw = ImageDraw.Draw(sheet)
        for k, f in enumerate(ranked[s:s + COLS * ROWS]):
            im = Image.open(resolve(index, f)).convert('RGB')
            im.thumbnail((CELL - 4, CELL - 4))
            x, y = (k % COLS) * CELL, (k // COLS) * (CELL + 14)
            sheet.paste(im, (x + 2, y + 2))
            draw.text((x + 3, y + CELL - 2), f'{s + k}  u{info[f]["lum_upper"]:.0f} w{len(info[f]["workers"])}',
                      fill=(0, 0, 0))
        sheet.save(f'{out}/sheets/sheet_{s // (COLS * ROWS):03d}.jpg', quality=85)
    json.dump([dict(rank=i, file=f, split=info[f]['split'], lum=info[f]['lum'], lum_upper=info[f]['lum_upper'],
                    n_workers=len(info[f]['workers'])) for i, f in enumerate(ranked)],
              open(f'{out}/ranked_darkest.json', 'w'), indent=0)
    print(len(info), 'annotated images;', len(ranked), 'darkest images on contact sheets in', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
