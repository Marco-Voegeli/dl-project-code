## Code for DL HS22 Project: Exploration of Structural-Textural Line Segment Classification
Team members: Yifan Yu, Shinjeong Kim, Alexis Tabin, Marco Voegeli

## Data Preparation and Processing
The code used to download, process, and generate normal difference values are included in `data` folder. The `csv` files are directly from [Hypersim](https://github.com/apple/ml-hypersim) dataset. `download.py` is modified from [this download script](https://github.com/apple/ml-hypersim/blob/main/contrib/99991) to download only the subset of data we used, with only RGB images, camera-world surface normals, and camera pose details (only for 3D line reconstruction).

`normal_diff_label.py` reads the downloaded data and produce data including the [LSD(Pytlsd)](https://github.com/iago-suarez/pytlsd) detections, ground truth filtering of the line segments (using 45deg threshold), and calculated surface normal differences per-pixel, which are then used to train our models.

## Omnidata Baseline
The Omnidata surface normal estimation baseline is easy to run using the demo in the [official repo](https://github.com/EPFL-VILAB/omnidata), the images need to be resized to 384x384, and the estimated surface normals are scaled back to original shape using nearest-neighbor interpolation. The surface normals are then processed using (commented) code in `data/normal_diff_label.py` to calculate the normal differences and classify the segments by comparing to threshold.

## Per-pixel Normal Difference Prediction

Instructions are in the [README](./normal-difference-prediction/README.md) file in `normal-difference-prediction`.

## Per-pixel Normal Difference Prediction (with weighted loss)

In `weighted-regression` we included a deviation from the prior model, where we apply a weight mask on the segments of the image where lines have been detected. Use `weighted-regression/regression.ipynb` to run, display the weight mask, and shows the performance of the model.
It requires the data generated from the data folder.

## End-to-End Detection and Classification with LETR
This adapted version of LETR allows simultaneous detection and classification of lines into two classes (structural, textural). It is modified from the code and model taken from this paper:
[Line Segment Detection Using Transformers without Edges](https://arxiv.org/abs/2101.01909) 

Follow instructions in the [README](./letr-adapted/README.md) file in `letr-adapted`.

## 3D Line Reconstruction
The 3D line reconstruction pipeline we used is one Yifan helped develop, but the pipeline is currently under review and is not open-sourced yet. So unfortunately we could not provide the code to reproduce the reconstruction results here.

In `reconstruction`, we included the reconstructed 3D line we put in the report in `.obj` files. The 3 files correspond to the reconstruction result by filter-before, filter-after, and using all line segments (baseline), on scene `ai_001_010` and `cam_00` trajectory. The lines kept are visible from at least 4 views.