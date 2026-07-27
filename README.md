# ERCAD

Official implementation of **ERCAD: An Embedding Replay Method for Continual Anomaly Detection and Segmentation**.

This repository provides the source code for ERCAD, a continual anomaly detection and segmentation framework based on embedding replay. The proposed method aims to address the catastrophic forgetting problem in continual anomaly detection scenarios.

## Installation

The required Python environment is provided in `requirements.txt`.

You can create the environment using:

```bash
pip install -r requirements.txt
Datasets

ERCAD is evaluated on two widely used industrial anomaly detection benchmarks:

MVTec AD
VisA
MVTec AD Dataset

MVTec AD is a widely used benchmark dataset for industrial anomaly detection and segmentation. It contains 15 categories of industrial objects and textures, including normal samples for training and various types of defective samples for testing. Pixel-level anomaly masks are provided for defective samples, enabling both image-level detection and pixel-level segmentation evaluation.

The dataset can be downloaded from:

https://www.mvtec.com/company/research/datasets/mvtec-ad

After downloading, please place the dataset according to the original MVTec AD directory structure:

MVTec_AD/
├── bottle/
│   ├── train/
│   ├── test/
│   └── ground_truth/
├── cable/
├── capsule/
├── carpet/
└── ...

The dataset path should be configured in run_MVTec.py before training.

VisA Dataset

VisA is a large-scale industrial anomaly detection dataset containing 12 object categories. It provides normal and anomalous images with pixel-level annotations for anomaly segmentation.

The dataset can be downloaded from:

https://github.com/amazon-science/spot-diff

The dataset should be organized as:

VisA/
├── candle/
├── capsules/
├── cashew/
├── chewinggum/
└── ...

The dataset path should be configured in run_Visa.py before training.

Training

ERCAD is designed for continual anomaly detection and segmentation. During training, anomaly detection tasks are learned sequentially. The embedding replay strategy is adopted to preserve previously learned anomaly knowledge while learning new tasks.

Training on MVTec AD

To train ERCAD on the MVTec AD dataset, run:

python run_MVTec.py

The training script will:

Load the MVTec AD dataset.
Construct the continual anomaly detection training process.
Optimize the ERCAD model.
Evaluate anomaly detection and segmentation performance.
Training on VisA

To train ERCAD on the VisA dataset, run:

python run_Visa.py

The training procedure follows the same continual learning setting as MVTec AD.
@article{deng_ercad_2026,
	title = {{ERCAD}: an embedding replay method for continual anomaly detection and segmentation},
	copyright = {All rights reserved},
	issn = {00313203},
	shorttitle = {Ercad},
	doi = {10.1016/j.patcog.2026.114507},
	journal = {Pattern Recognition},
	author = {Deng, Zhipeng and Yang, Gen and Tu, Bing and Liu, Yong and Man, Junfeng},
	month = jul,
	year = {2026},
	pages = {114507}
}
