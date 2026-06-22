# Benchmarking Deep Segmentation Architectures and Data Preparation Strategies for Echocardiographic Image Segmentation

This is the official repository for our paper:

**Benchmarking Deep Segmentation Architectures and Data Preparation Strategies for Echocardiographic Image Segmentation**

## Overview

Accurate cardiac chamber segmentation is essential for estimating left ventricular volumes and ejection fraction from echocardiography. This repository provides a unified benchmark of three influential deep learning architectures:

- U-Net
- Attention U-Net
- TransUNet

The study investigates how model architecture, preprocessing strategy, self-supervised learning, and SAM-based pseudo-labeling influence segmentation performance on cardiac ultrasound images.

---

## Key Contributions

- Unified benchmark of U-Net, Attention U-Net, and TransUNet.
- Analysis of NIfTI versus 16-bit PNG preprocessing pipelines.
- SimCLR-based self-supervised pretraining using unlabeled echocardiographic frames.
- SAM-assisted pseudo-label generation for semi-supervised learning.
- Reproducible evaluation on the CAMUS dataset.

---

## Framework







The framework combines:

1. Literature review and benchmark design
2. Data preprocessing and quality assurance
3. Self-supervised representation learning
4. SAM-based pseudo-label generation
5. Cardiac segmentation using U-Net, Attention U-Net, and TransUNet


---

## Dataset

### CAMUS Dataset

The benchmark uses the CAMUS dataset:

- 500 patients
- Apical 2-chamber and 4-chamber views
- End-diastole (ED) and End-systole (ES) annotations
- LV Endocardium
- LV Myocardium
- Left Atrium

Dataset Website:

https://www.creatis.insa-lyon.fr/Challenge/camus/

---

## Network Architectures

### U-Net

Classical encoder-decoder architecture with skip connections.

### Attention U-Net

Extends U-Net with attention gates to suppress irrelevant background regions.

### TransUNet

Hybrid CNN-Transformer architecture for capturing both local and global contextual information.

---

## Self-Supervised Learning

We employ SimCLR-based pretraining on approximately 43,000 unlabeled echocardiographic frames before supervised fine-tuning.

---

## SAM-Based Pseudo Labeling

The Segment Anything Model (SAM) is used to generate pseudo-labels from unlabeled echocardiographic frames.

Filtering criteria:

- Predicted IoU ≥ 0.7
- Minimum Area ≥ 200 pixels
- Top-3 mask selection

---


## Experimental Results



## Installation

```bash
https://github.com/Zahid672/CAMUS_public.git

cd CAMUS_public


pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Evaluation

```bash
python evaluate.py
```

## Repository Structure

```text
├── Data/
├── Figures/
├── Models/
├── Scripts/
├── Results/
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Citation
