# ERCAD

Official implementation of **ERCAD: An Embedding Replay Method for Continual Anomaly Detection and Segmentation**.

This repository provides the source code for ERCAD, a continual anomaly detection and segmentation framework based on embedding replay. The proposed method aims to address the catastrophic forgetting problem in continual anomaly detection scenarios.

## Installation

The required Python environment is provided in `requirements.txt`.

You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Datasets

ERCAD is evaluated on two widely used industrial anomaly detection benchmarks:

- **MVTec AD**
- **VisA**

### MVTec AD Dataset

MVTec AD is a widely used benchmark dataset for industrial anomaly detection and segmentation. It contains 15 categories of industrial objects and textures, including normal samples for training and various types of defective samples for testing. Pixel-level anomaly masks are provided for defective samples, enabling both image-level anomaly detection and pixel-level anomaly segmentation evaluation.

The dataset can be downloaded from:

https://www.mvtec.com/company/research/datasets/mvtec-ad

After downloading, please keep the original MVTec AD dataset structure:

```text
MVTec_AD/
├── bottle/
│   ├── train/
│   ├── test/
│   └── ground_truth/
├── cable/
├── capsule/
├── carpet/
└── ...
```

Please configure the dataset path in `run_MVTec.py` before training.

### VisA Dataset

VisA is a large-scale industrial anomaly detection dataset containing 12 object categories. It provides normal and anomalous images with pixel-level annotations for anomaly segmentation.

The dataset can be downloaded from:

https://github.com/amazon-science/spot-diff

After downloading, organize the dataset as follows:

```text
VisA/
├── candle/
├── capsules/
├── cashew/
├── chewinggum/
└── ...
```

Please configure the dataset path in `run_Visa.py` before training.

## Training

ERCAD is designed for continual anomaly detection and segmentation. During training, anomaly detection tasks are learned sequentially. The embedding replay strategy is adopted to preserve previously learned anomaly representations while learning new tasks.

### Training on MVTec AD

To train ERCAD on the MVTec AD dataset, run:

```bash
python run_MVTec.py
```

The training script will:

1. Load the MVTec AD dataset.
2. Construct the continual anomaly detection training process.
3. Optimize the ERCAD model.
4. Evaluate anomaly detection and segmentation performance.

### Training on VisA

To train ERCAD on the VisA dataset, run:

```bash
python run_Visa.py
```

The training procedure follows the same continual learning setting as MVTec AD.

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{deng_ercad_2026,
	title = {{ERCAD}: an embedding replay method for continual anomaly detection and segmentation},
	doi = {10.1016/j.patcog.2026.114507},
	journal = {Pattern Recognition},
	author = {Deng, Zhipeng and Yang, Gen and Tu, Bing and Liu, Yong and Man, Junfeng},
	month = jul,
	year = {2026},
	pages = {114507}
}
```
