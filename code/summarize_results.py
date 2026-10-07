"""Recompute the result tables of the paper from the per-run test metrics in results/.

Table 2  : P, R, F1, mAP@0.5 and mAP@0.5:0.95 of the four detectors on the three test sets (mean and standard deviation
           over seeds 0, 1 and 2) and the relative changes between detectors, computed from unrounded means
Table 3  : class-level mAP@0.5 of the four detectors
Table S1 : the same metrics with a 50-epoch schedule for all detectors and the urban CycleGAN

Usage: python summarize_results.py <results folder>
Output: <results folder>/summary/table_2.csv, table_3.csv and table_s1.csv
"""
import csv
import json
import statistics
import sys
from pathlib import Path

TESTS = ['ppe_test', 'sfchd_day', 'sfchd_night']
METRICS = ['P', 'R', 'F1', 'mAP50', 'mAP50_95']


def load(folder, name):
    runs = [json.loads(Path(folder, f'{name}_s{s}.json').read_text()) for s in range(3)]
    return runs


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def change(a, b):
    return (a / b - 1) * 100


def results_table(folder, detectors, changes, path):
    data = {name: load(folder, name) for name, _ in detectors}
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['test_set', 'detector'] + [f'{m}_mean' for m in METRICS] + [f'{m}_sd' for m in METRICS])
        for t in TESTS:
            means = {}
            for name, label in detectors:
                stats = [mean_sd([r[t][m] for r in data[name]]) for m in METRICS]
                means[name] = [s[0] for s in stats]
                w.writerow([t, label] + [f'{s[0]:.4f}' for s in stats] + [f'{s[1]:.4f}' for s in stats])
            for label, a, b in changes:
                w.writerow([t, label] + [f'{change(x, y):+.1f}%' for x, y in zip(means[a], means[b])] + [''] * len(METRICS))
    return data


def main(results):
    results = Path(results)
    out = results / 'summary'
    out.mkdir(exist_ok=True)
    detectors = [('baseline', 'Baseline Model'), ('sizematch', 'Size-matched control'),
                 ('enhanced_urban', 'Enhanced Model, urban CycleGAN'),
                 ('enhanced_construction_night', 'Enhanced Model, construction-night CycleGAN')]
    changes = [('Change, construction-night Enhanced vs Baseline', 'enhanced_construction_night', 'baseline'),
               ('Change, construction-night Enhanced vs control', 'enhanced_construction_night', 'sizematch'),
               ('Change, urban Enhanced vs control', 'enhanced_urban', 'sizematch'),
               ('Change, construction-night vs urban Enhanced', 'enhanced_construction_night', 'enhanced_urban')]
    data = results_table(results / 'test_metrics', detectors, changes, out / 'table_2.csv')
    with open(out / 'table_3.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['test_set', 'class'] + [label for _, label in detectors] +
                   ['Change, construction-night Enhanced vs Baseline', 'Change, construction-night Enhanced vs control'])
        for t in TESTS:
            classes = list(data['baseline'][0][t]['per_class'])
            for c in classes:
                means = {n: statistics.mean(r[t]['per_class'][c]['mAP50'] for r in data[n]) for n, _ in detectors}
                w.writerow([t, c] + [f'{means[n]:.4f}' for n, _ in detectors] +
                           [f"{change(means['enhanced_construction_night'], means['baseline']):+.1f}%",
                            f"{change(means['enhanced_construction_night'], means['sizematch']):+.1f}%"])
    results_table(results / 'supplementary_table_s1_50_epochs' / 'test_metrics',
                  [('baseline', 'Baseline Model'), ('sizematch', 'Size-matched control'),
                   ('enhanced_urban', 'Enhanced Model, urban CycleGAN')],
                  [('Change, control vs Baseline', 'sizematch', 'baseline'),
                   ('Change, urban Enhanced vs Baseline', 'enhanced_urban', 'baseline'),
                   ('Change, urban Enhanced vs control', 'enhanced_urban', 'sizematch')],
                  out / 'table_s1.csv')
    print('written', sorted(p.name for p in out.iterdir()))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'results')
