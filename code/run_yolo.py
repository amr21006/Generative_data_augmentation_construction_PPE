"""Controlled YOLOv8s experiment: Baseline vs Enhanced (CycleGAN night) vs Size-matched control.

Arms (same Construction-PPE val split for checkpoint selection, same hyperparameters):
  baseline  : Construction-PPE training images (n = 1,132)
  enhanced  : training images + their CycleGAN day-to-night translations (n = 2,264)
  sizematch : training images listed twice (n = 2,264), to control for training-set size
Test sets (never used for training or checkpoint selection):
  ppe_test    : Construction-PPE test split (141 images, 11 classes)
  sfchd_day   : 447 daytime frames, SFCHD outdoor cameras (Person, helmet)
  sfchd_night : 447 night-time frames, same cameras (Person, helmet)
Outputs under --out: yolo_runs/<arm>_s<seed>/ (results.csv, weights, val outputs) and results/<arm>_s<seed>.json
"""
import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path

NAMES = ['helmet', 'gloves', 'vest', 'boots', 'goggles', 'none', 'Person',
         'no_helmet', 'no_goggle', 'no_gloves', 'no_boots']
HYP = dict(epochs=50, imgsz=640, optimizer='AdamW', lr0=0.001, weight_decay=0.0005,
           cos_lr=True, warmup_epochs=3, warmup_momentum=0.8, deterministic=False)


def link(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare(data, synth, work, arm):
    """Build the training list for one arm (each image a distinct file with its own label file)."""
    data, work = Path(data), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in (data / 'images/train').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    base = [str(p) for p in imgs]
    extra = []
    if arm in ('enhanced', 'sizematch'):
        tag = 'night' if arm == 'enhanced' else 'dup'
        (work / f'{tag}/images').mkdir(parents=True, exist_ok=True)
        (work / f'{tag}/labels').mkdir(parents=True, exist_ok=True)
        missing = []
        for p in imgs:
            lab = data / 'labels/train' / (p.stem + '.txt')
            if tag == 'night':
                s = Path(synth) / (p.stem + '_n2.jpg')
                if not s.exists():
                    missing.append(s.name)
                    continue
                link(s, work / f'night/images/{p.stem}_n2.jpg')
                link(lab, work / f'night/labels/{p.stem}_n2.txt')
                extra.append(str(work / f'night/images/{p.stem}_n2.jpg'))
            else:
                link(p, work / f'dup/images/{p.stem}_dup{p.suffix}')
                link(lab, work / f'dup/labels/{p.stem}_dup.txt')
                extra.append(str(work / f'dup/images/{p.stem}_dup{p.suffix}'))
        if missing:
            raise FileNotFoundError(f'{len(missing)} translated images missing, e.g. {missing[:3]}')
    L = base + extra
    (work / f'{arm}.txt').write_text(chr(10).join(L) + chr(10))
    yaml = {'path': str(work), 'train': str(work / f'{arm}.txt'),
            'val': str(data / 'images/val'), 'names': dict(enumerate(NAMES))}
    (work / f'{arm}.yaml').write_text(json.dumps(yaml))  # JSON is valid YAML
    print(arm, len(L), 'training images', flush=True)
    return work / f'{arm}.yaml'


def test_yamls(data, sfchd_root, work, tag, extra=()):
    data, sfchd_root, work = Path(data), Path(sfchd_root), Path(work)
    out = {}
    sets = [('ppe_test', data / 'images/test'),
            ('sfchd_day', sfchd_root / 'sfchd_day/images'),
            ('sfchd_night', sfchd_root / 'sfchd_night/images')]
    for e in extra:   # additional test sets 'name=folder' (folder has images/ and labels/ in Construction-PPE ids)
        n, _, p = e.partition('=')
        sets.append((n, Path(p) / 'images'))
    for name, img in sets:
        if not img.exists():
            raise FileNotFoundError(img)
        y = work / f'test_{name}_{tag}.yaml'
        y.write_text(json.dumps({'path': str(img.parent), 'train': str(img), 'val': str(img),
                                 'names': dict(enumerate(NAMES))}))
        out[name] = y
    return out


def metrics_dict(m):
    b = m.box
    per = {}
    for i, c in enumerate(b.ap_class_index):
        per[NAMES[int(c)]] = dict(P=float(b.p[i]), R=float(b.r[i]), F1=float(b.f1[i]),
                                  mAP50=float(b.ap50[i]), mAP50_95=float(b.ap[i]))
    P, R = float(b.mp), float(b.mr)
    return dict(P=P, R=R, F1=(2 * P * R / (P + R) if P + R else 0.0), mAP50=float(b.map50),
                mAP50_95=float(b.map), per_class=per, speed=m.speed)


def env_info():
    import torch
    import ultralytics
    return dict(python=platform.python_version(), torch=torch.__version__, ultralytics=ultralytics.__version__,
                cuda=torch.version.cuda, gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Construction-PPE root (images/, labels/)')
    ap.add_argument('--synth', default='', help='folder of CycleGAN night translations (*_n2.jpg)')
    ap.add_argument('--sfchd', required=True, help='folder containing sfchd_day/ and sfchd_night/')
    ap.add_argument('--out', default='.')
    ap.add_argument('--work', default=None)
    ap.add_argument('--arms', default='baseline,enhanced,sizematch')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--amp', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=HYP['epochs'])
    ap.add_argument('--fraction', type=float, default=1.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='0')
    ap.add_argument('--cache', default='ram')
    ap.add_argument('--prepare_only', action='store_true')
    ap.add_argument('--extra', action='append', default=[], help='extra test set name=folder')
    a = ap.parse_args()
    out = Path(a.out).resolve()
    work = Path(a.work) if a.work else out / 'yolo_data'
    if a.prepare_only:
        for arm in a.arms.split(','):
            prepare(a.data, a.synth, work, arm)
        return
    from ultralytics import YOLO
    res_dir = out / 'results'
    res_dir.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in a.seeds.split(',')]:
        for arm in a.arms.split(','):
            name = f'{arm}_s{seed}{a.tag}'
            out_json = res_dir / f'{name}.json'
            if out_json.exists():
                print('skip', name, flush=True)
                continue
            t0 = time.time()
            data_yaml = prepare(a.data, a.synth, work, arm)
            tests = test_yamls(a.data, a.sfchd, work, name, a.extra)
            model = YOLO('yolov8s.pt')
            hyp = dict(HYP, epochs=a.epochs)
            model.train(data=str(data_yaml), batch=a.batch, workers=a.workers,
                        seed=seed, amp=bool(a.amp), project=str(out / 'yolo_runs'), name=name, device=a.device,
                        cache=(a.cache if a.cache != 'none' else False),
                        exist_ok=True, fraction=a.fraction, plots=False, verbose=False, **hyp)
            train_min = (time.time() - t0) / 60
            best = out / 'yolo_runs' / name / 'weights' / 'best.pt'
            ev = {}
            m = YOLO(str(best))
            for tname, y in tests.items():
                r = m.val(data=str(y), split='val', imgsz=640, batch=a.batch, conf=0.001, iou=0.7,
                          device=a.device, plots=False, save_json=True, verbose=False,
                          project=str(out / 'yolo_runs' / name), name=f'val_{tname}', exist_ok=True)
                ev[tname] = metrics_dict(r)
            ev.update(train_minutes=train_min, total_minutes=(time.time() - t0) / 60, arm=arm, seed=seed,
                      batch=a.batch, amp=bool(a.amp), epochs=a.epochs, env=env_info())
            out_json.write_text(json.dumps(ev, indent=1))
            print(name, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk in ('mAP50', 'mAP50_95', 'P', 'R')}
                                    for k, v in ev.items() if isinstance(v, dict) and 'mAP50' in v}), flush=True)


if __name__ == '__main__':
    main()
