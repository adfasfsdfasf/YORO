# ERCAD

Official implementation of:

**ERCAD: An Embedding Replay Method for Continual Anomaly Detection and Segmentation**

This repository contains the official PyTorch implementation of ERCAD for continual anomaly detection and segmentation.

## 📌 Overview

ERCAD is an embedding replay based framework designed to alleviate catastrophic forgetting in continual anomaly detection scenarios.

## 🔧 Installation

Clone this repository:

```bash
git clone https://github.com/xxx/ERCAD.git
cd ERCAD
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📂 Dataset Preparation

ERCAD is evaluated on two industrial anomaly detection benchmarks:

- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [VisA](https://github.com/amazon-science/spot-diff)

### MVTec AD

Download the MVTec AD dataset and place it as follows:

```
datasets/
└── MVTec_AD/
    ├── bottle/
    ├── cable/
    ├── capsule/
    ├── carpet/
    └── ...
```

The dataset path should be specified in:

```
run_MVTec.py
```

before training.

### VisA

Download the VisA dataset and organize it as:

```
datasets/
└── VisA/
    ├── candle/
    ├── capsules/
    ├── cashew/
    ├── chewinggum/
    └── ...
```

The dataset path should be specified in:

```
run_Visa.py
```

before training.

## 🚀 Training

### Train on MVTec AD

Run:

```bash
python run_MVTec.py
```

The script will automatically perform:

- dataset loading
- continual anomaly detection training
- model optimization
- anomaly detection and segmentation evaluation


### Train on VisA

Run:

```bash
python run_Visa.py
```

The training procedure follows the same continual learning setting as MVTec AD.

## 📊 Evaluation

ERCAD evaluates both image-level anomaly detection and pixel-level anomaly segmentation.

The commonly used metrics include:

- Image-level AUROC
- Pixel-level AUROC
- PRO score


## 📄 Citation

If you find this work useful, please cite:

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

## 📬 Contact

For questions or discussions, please contact the authors.
