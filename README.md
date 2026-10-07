# Generative day-to-night data augmentation for construction PPE detection

This repository accompanies the article "A Generative Data Augmentation Framework for Low-Light Object Detection and
Automated Safety Surveillance on Construction Sites". It contains the code, the image lists, the logged results and the
day-to-night generator of the urban CycleGAN, so that every number in the article can be traced to a logged run and the
study can be repeated.

## Study design

1. **Day-to-night translation.** Two CycleGANs with the same configuration translate the 1,132 Construction-PPE training
   images into night-time images; each translated image reuses the labels of its source image.
   - *Urban CycleGAN:* 522 daytime and 227 night-time urban photographs (Unpaired Day and Night Cityview Images).
   - *Construction-night CycleGAN:* the 1,013 Construction-PPE training images in JPEG format as the daytime domain and
     205 night-time construction-site photographs selected from MOCS as the night-time domain.
2. **Detector training.** YOLOv8s detectors, each trained with seeds 0, 1 and 2:
   - *Baseline Model:* the 1,132 original images, 50 epochs.
   - *Size-matched control:* two copies of each original image (2,264 images), 25 epochs.
   - *Enhanced Model (urban or construction-night):* the original images and their 1,132 translations (2,264 images),
     25 epochs.

   The update-matched schedule (Equation 5 of the article) gives every detector the same 56,600 processed training
   images; the warm-up and the final epochs without mosaic augmentation are scaled in the same way (1.5 and 5 epochs for
   25-epoch runs, 3 and 10 epochs for 50-epoch runs).
3. **Configuration selection.** Four configurations of the construction-night Enhanced Model were compared on a
   night-time validation set of 88 MOCS photographs with 171 annotated workers (Supplementary Table S2).
4. **Evaluation.** Every detector is tested on the Construction-PPE test split (141 images, 11 classes) and on 447
   daytime and 447 night-time frames from the outdoor cameras of SFCHD (Person and helmet).

## Contents

| Path | Content |
|---|---|
| `code/cyclegan_d2n.py` | CycleGAN training (`train`), translation of a folder (`translate`), example panels (`panels`) and luminance statistics (`stats`). |
| `code/run_yolo.py` | Builds the training sets, trains YOLOv8s and evaluates each detector on the test sets (or, with `--only_extra`, on the validation set only). |
| `code/scan_mocs_night.py` | Ranks the annotated MOCS images by luminance and draws numbered contact sheets of the darkest images for inspection. |
| `code/select_mocs_night.py` | Applies the inspection decisions, removes near-identical frames and splits the night-time images into the translation pool and the validation set. |
| `code/prepare_construction_night_data.py` | Builds the training folders of the construction-night CycleGAN and the YOLO-format validation set. |
| `code/build_sfchd_testsets.py` | Builds the SFCHD daytime and night-time test sets from the public SFCHD release and the image lists. |
| `code/summarize_results.py` | Recomputes Tables 2, 3 and S1 of the article from the per-run metrics in `results/`. |
| `file_lists/` | Construction-PPE splits, the 1,013 JPEG training images used as the daytime domain, the 560 inspected MOCS candidates with their decisions (`mocs_night_candidates.csv`), the 205 translation-pool and 88 validation images, and the SFCHD test frames. |
| `results/test_metrics/` | Test metrics of the twelve detectors of Tables 2 and 3 (precision, recall, F1-score, mAP@0.5, mAP@0.5:0.95 and class-level mAP@0.5), with their training settings. |
| `results/training_curves/` | Per-epoch training and validation curves of these detectors (Figure 5). |
| `results/supplementary_table_s1_50_epochs/` | Metrics and curves of the 50-epoch schedule (Supplementary Table S1). |
| `results/supplementary_table_s2_configuration_selection/` | Validation results of the configuration selection (Supplementary Table S2). |
| `results/cyclegan/` | Training logs of both CycleGANs (Figure 4), sample sheets of the kept epochs (input, translation and reconstruction) and the translation time. |
| `results/summary/` | Tables 2, 3 and S1 recomputed by `code/summarize_results.py`. |
| `weights/urban_cyclegan_generator.pt` | Day-to-night generator of the urban CycleGAN (weights of epoch 49). |

## Data sources

- **Construction-PPE** (Dalvi et al., 2025): <https://docs.ultralytics.com/datasets/detect/construction-ppe/> (AGPL-3.0).
- **Unpaired Day and Night Cityview Images** (heonh0, 2022): <https://www.kaggle.com/datasets/heonh0/daynight-cityview>.
- **MOCS** (An et al., 2021, doi: 10.1016/j.autcon.2020.103482): public release of the annotated images, for example
  <https://www.kaggle.com/datasets/xiaopan9802/mocs-dataset>. Use under the terms of the dataset.
- **SFCHD** (Yu et al., 2023): <https://arxiv.org/abs/2306.02098>. Obtain the images from the dataset authors.

Translated images derived from Construction-PPE are subject to its licence.

## Reproducing the study

Software: Python 3.13, PyTorch 2.11, Ultralytics 8.4.82; one NVIDIA Tesla T4 GPU per run (the 50-epoch comparison of
Supplementary Table S1 used PyTorch 2.10). `<PPE>` is the Construction-PPE root with `images/` and `labels/`.

1. **Test sets.** `python code/build_sfchd_testsets.py <SFCHD folder> file_lists data/sfchd`
2. **MOCS night-time images.**
   ```
   python code/scan_mocs_night.py <MOCS root> data/mocs_scan
   python code/select_mocs_night.py --scan data/mocs_scan --accept "<ranks>" --reject "<ranks>" --out data/mocs_night
   python code/prepare_construction_night_data.py <PPE> <MOCS root> data/mocs_night/night_split.json data/construction_night
   ```
   The accepted and rejected ranks of the study are given in the header of `select_mocs_night.py`; with them, the script
   reproduces the image lists in `file_lists/`.
3. **CycleGANs and translation.**
   ```
   python code/cyclegan_d2n.py train --data <Cityview root with day/ and night/> --out cyclegan_urban --epochs 50 --steps 200 --bs 2 --crop 256
   python code/cyclegan_d2n.py train --data data/construction_night/cyclegan_data --out cyclegan_construction_night --epochs 50 --steps 200 --bs 2 --crop 256
   python code/cyclegan_d2n.py translate --weights cyclegan_urban/best.pt --src <PPE>/images/train --dst data/translated_urban --long_side 640
   python code/cyclegan_d2n.py translate --weights cyclegan_construction_night/best.pt --src <PPE>/images/train --dst data/translated_construction_night --long_side 640
   ```
   With `--crop 256`, images are resized to 350 x 600 pixels before cropping. The released urban generator can be used
   directly with `translate --weights weights/urban_cyclegan_generator.pt`.
4. **Detectors (Tables 2 and 3).**
   ```
   python code/run_yolo.py --data <PPE> --sfchd data/sfchd --arms baseline --seeds 0,1,2 --batch 32 --epochs 50 --warmup_epochs 3 --close_mosaic 10 --out runs
   python code/run_yolo.py --data <PPE> --sfchd data/sfchd --arms sizematch --seeds 0,1,2 --batch 32 --epochs 25 --warmup_epochs 1.5 --close_mosaic 5 --out runs
   python code/run_yolo.py --data <PPE> --sfchd data/sfchd --synth data/translated_urban --arms enhanced --seeds 0,1,2 --batch 32 --epochs 25 --warmup_epochs 1.5 --close_mosaic 5 --tag _urban --out runs
   python code/run_yolo.py --data <PPE> --sfchd data/sfchd --synth data/translated_construction_night --arms enhanced --seeds 0,1,2 --batch 32 --epochs 25 --warmup_epochs 1.5 --close_mosaic 5 --tag _construction_night --out runs
   ```
5. **Supplementary Table S1** (50 epochs for all detectors, urban CycleGAN):
   `python code/run_yolo.py --data <PPE> --sfchd data/sfchd --synth data/translated_urban --arms baseline,sizematch,enhanced --seeds 0,1,2 --batch 32 --epochs 50 --out runs_50`
6. **Supplementary Table S2** (validation set only, seed 0): run `run_yolo.py` with `--seeds 0 --only_extra --extra night_val=data/construction_night/night_val`
   for each configuration (25 or 50 epochs with all translations; 33 or 50 epochs with `--synth_share 0.5`).
7. **Tables.** `python code/summarize_results.py results`

GPU training is not bit-for-bit deterministic, so retrained models reproduce the procedure and the order of magnitude of
the results rather than identical values; the logged results of the article are in `results/`.
