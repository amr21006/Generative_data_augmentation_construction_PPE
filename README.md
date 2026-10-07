# Generative day-to-night data augmentation for construction PPE detection

This repository accompanies the study "A Generative Data Augmentation Framework for Low-Light Object Detection and Automated Safety Surveillance on Construction Sites". It contains the code, the trained models, the translated night-time training images, the logged results and the file lists needed to reproduce the experiment.

## Experiment

1. A CycleGAN is trained on unpaired daytime (522) and night-time (227) urban photographs.
2. The day-to-night generator translates the 1,132 Construction-PPE training images; each translated image reuses the YOLO labels of its source image.
3. YOLOv8s detectors are trained on three training sets, each with seeds 0, 1 and 2:
   - **Baseline:** the 1,132 original images.
   - **Enhanced:** the original and the translated images (2,264).
   - **Size-matched control:** the original images listed twice (2,264).
4. Every detector is tested on the Construction-PPE test split (141 images) and on 447 daytime and 447 night-time SFCHD frames (Person and helmet).

## Contents

| Folder | Content |
|---|---|
| `code/run_all_kaggle.ipynb` | Runs the complete experiment in one Kaggle notebook (GPU T4 x2): CycleGAN training, translation, training of the nine detectors and evaluation. |
| `code/cyclegan_d2n.py` | CycleGAN training (`train`), translation of a folder (`translate`), example panels (`panels`) and luminance statistics (`stats`). |
| `code/run_yolo.py` | Builds the three training sets, trains YOLOv8s and evaluates each detector on the three test sets. |
| `code/build_sfchd_testsets.py` | Builds the SFCHD daytime and night-time test sets from the public SFCHD release and the file lists. |
| `code/summarize_results.py` | Mean ± standard deviation over seeds and relative changes between detectors. |
| `weights/cyclegan/cyclegan_day2night_generator.pt` | Day-to-night generator G (epoch 49 of 50). |
| `weights/yolov8s/` | Best checkpoint (validation mAP@0.5:0.95) of each detector: `baseline_s{0,1,2}.pt`, `sizematch_s{0,1,2}.pt`, `enhanced_s{0,1,2}.pt`. |
| `construction_ppe_night_translated/` | The 1,132 translated training images (`images/*_n2.jpg`) with the labels of their source images (`labels/*_n2.txt`). These synthetic images are for training only. |
| `results/` | Test metrics of each detector (`results/<arm>_s<seed>.json`), per-epoch validation curves (`training_curves/`), CycleGAN training log, luminance statistics and software environment. |
| `file_lists/` | Image names of the Construction-PPE train, validation and test splits, the SFCHD test frames and the translated images. |

## Data sources

- **Construction-PPE** (Dalvi et al., 2025): <https://docs.ultralytics.com/datasets/detect/construction-ppe/> (AGPL-3.0). The notebook downloads it automatically.
- **Unpaired Day and Night Cityview Images** (heonh0, 2022): <https://www.kaggle.com/datasets/heonh0/daynight-cityview> (CC0).
- **SFCHD** (Yu et al., 2023): <https://arxiv.org/abs/2306.02098>. Obtain the images from the dataset authors and build the test sets with `code/build_sfchd_testsets.py`.

The translated images are derived from Construction-PPE and are distributed under its licence (AGPL-3.0).

## Reproducing the results

1. Build the SFCHD test sets:

   ```
   python code/build_sfchd_testsets.py <SFCHD folder> file_lists <output folder>
   ```

   Upload the output folder as a Kaggle dataset.
2. Open `code/run_all_kaggle.ipynb` on Kaggle with Accelerator = GPU T4 x2 and Internet on.
3. Add two inputs: the dataset `heonh0/daynight-cityview` and the SFCHD test-set dataset.
4. Run all cells. The notebook writes `outputs.zip` (results and logs) and `weights.zip` (trained models).
5. Summarize the results:

   ```
   python code/summarize_results.py <unzipped outputs folder>
   ```

   This prints the mean ± standard deviation and the relative changes for every test set.

Software: Python 3.12, PyTorch 2.10, Ultralytics 8.4.82.
