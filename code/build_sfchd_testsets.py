"""Build the SFCHD daytime and night-time test sets used in the study from the public SFCHD release.

SFCHD (Yu et al., 2023; https://arxiv.org/abs/2306.02098) is distributed with YOLO labels. The frames used here are listed
in file_lists/sfchd_night.txt (447 night-time frames from ten outdoor cameras) and file_lists/sfchd_day.txt (447 daytime
frames from the same cameras, drawn at random with seed 2026 and matched to the night-time set by camera).
Only the classes with a Construction-PPE counterpart are kept: SFCHD person (0) -> Person (6) and SFCHD helmet (1) -> helmet (0).

Usage: python build_sfchd_testsets.py <SFCHD folder with images/ and labels/> <file_lists folder> <output folder>
Output: <output>/sfchd_day/{images,labels} and <output>/sfchd_night/{images,labels}
"""
import collections
import shutil
import sys
from pathlib import Path

MAP = {0: 6, 1: 0}   # SFCHD person -> Person (6); SFCHD helmet -> helmet (0)


def build(src, lists, out):
    src, lists, out = Path(src), Path(lists), Path(out)
    for name in ('sfchd_night', 'sfchd_day'):
        files = [l.strip() for l in (lists / f'{name}.txt').read_text().splitlines() if l.strip()]
        (out / name / 'images').mkdir(parents=True, exist_ok=True)
        (out / name / 'labels').mkdir(parents=True, exist_ok=True)
        cnt = collections.Counter()
        for f in files:
            shutil.copy(src / 'images' / f, out / name / 'images' / f)
            lines = []
            lp = src / 'labels' / (Path(f).stem + '.txt')
            if lp.exists():
                for line in lp.read_text().splitlines():
                    p = line.split()
                    if len(p) == 5 and int(p[0]) in MAP:
                        lines.append(' '.join([str(MAP[int(p[0])])] + p[1:]))
                        cnt[MAP[int(p[0])]] += 1
            (out / name / 'labels' / (Path(f).stem + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''))
        print(name, len(files), 'images |', cnt[6], 'Person |', cnt[0], 'helmet')


if __name__ == '__main__':
    build(*sys.argv[1:4])
