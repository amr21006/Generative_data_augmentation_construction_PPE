"""Prepare the training folders of the construction-night CycleGAN and the MOCS night-time validation set.

CycleGAN domains (read by cyclegan_d2n.py train --data <out>/cyclegan_data):
  day/   the Construction-PPE training images in JPEG format (1,013 images; file_lists/construction_ppe_train_jpeg.txt)
  night/ the 205 MOCS night-time photographs of the translation pool (file_lists/mocs_night_translation_pool.txt)
Validation set (YOLO format, used with run_yolo.py --only_extra --extra night_val=<out>/night_val):
  night_val/images, night_val/labels  the 88 MOCS validation photographs; MOCS workers are written as the
                                      Construction-PPE class Person (class index 6)

Usage: python prepare_construction_night_data.py <Construction-PPE root> <MOCS root> <night_split.json> <out>
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path


def copy(src, dst):
    if not Path(dst).exists():
        shutil.copy2(src, dst)


def main(ppe, mocs, split_json, out):
    split = json.loads(Path(split_json).read_text())
    index = {}
    for p in glob.glob(f'{mocs}/**/*.jpg', recursive=True):
        index.setdefault(os.path.basename(p), []).append(p)

    def mocs_path(key):
        sp, fn = key.split('/', 1)
        c = index.get(os.path.basename(fn), [])
        inside = [q for q in c if sp in q.replace(os.sep, '/')]
        return inside[0] if inside else c[0]

    for d in ('cyclegan_data/day', 'cyclegan_data/night', 'night_val/images', 'night_val/labels'):
        Path(out, d).mkdir(parents=True, exist_ok=True)
    for p in sorted(glob.glob(f'{ppe}/images/train/*.jpg')):
        copy(p, Path(out, 'cyclegan_data/day', os.path.basename(p)))
    for f in split['pool']:
        copy(mocs_path(f), Path(out, 'cyclegan_data/night', f.replace('/', '__')))
    n_boxes = 0
    for f, r in split['val'].items():
        name = f.replace('/', '__')
        copy(mocs_path(f), Path(out, 'night_val/images', name))
        w_img, h_img = r['wh']
        lines = [f'6 {(x + w / 2) / w_img:.6f} {(y + h / 2) / h_img:.6f} {w / w_img:.6f} {h / h_img:.6f}'
                 for x, y, w, h in r['workers']]
        n_boxes += len(lines)
        Path(out, 'night_val/labels', Path(name).stem + '.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
    print('day', len(os.listdir(Path(out, 'cyclegan_data/day'))), '| night', len(os.listdir(Path(out, 'cyclegan_data/night'))),
          '| validation', len(split['val']), 'images,', n_boxes, 'worker boxes')


if __name__ == '__main__':
    main(*sys.argv[1:5])
