"""Summarize the nine YOLO runs: mean ± SD per arm and test set, and relative changes between arms."""
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'kaggle_outputs')
R = {p.stem: json.loads(p.read_text()) for p in sorted((root / 'results').glob('*_s[0-9].json'))}
sets = ['ppe_test', 'sfchd_day', 'sfchd_night']
print('run            ' + ' | '.join(f'{s:>22s}' for s in sets), '| train_min')
for k, r in R.items():
    print(f'{k:14s} ' + ' | '.join(f"{r[s]['mAP50']:.3f} / {r[s]['mAP50_95']:.3f}".rjust(22) for s in sets),
          f"| {r['train_minutes']:.1f}")
for metric in ('mAP50', 'mAP50_95', 'F1', 'P', 'R'):
    print('\n==', metric)
    M = {}
    for arm in ('baseline', 'sizematch', 'enhanced'):
        M[arm] = {s: [R[f'{arm}_s{i}'][s][metric] for i in range(3)] for s in sets}
        print(f'  {arm:10s}', '  '.join(f'{s}: {st.mean(v):.4f} ± {st.stdev(v):.4f}' for s, v in M[arm].items()))
    for a, b in (('enhanced', 'baseline'), ('enhanced', 'sizematch'), ('sizematch', 'baseline')):
        print(f'  {a} vs {b}: ',
              '  '.join(f'{s}: {(st.mean(M[a][s]) - st.mean(M[b][s])) / st.mean(M[b][s]) * 100:+.1f}%' for s in sets))
print('\nper-seed wins, enhanced vs sizematch (mAP50):',
      {s: sum(R[f'enhanced_s{i}'][s]['mAP50'] > R[f'sizematch_s{i}'][s]['mAP50'] for i in range(3)) for s in sets})
