## First things first! Pretrained models can be found here:
https://drive.google.com/drive/folders/1aWNdRHJqgsHlvnzahH66naxPWpQ-hzuQ?usp=sharing

* Due to high volume of the trained models, I have shared them in a google drive link! Please feel free to have a look and use the models for your work, in that case I would be glad if you please let me know about it, yet there are no licensing restrictions here, all is my independent work!

# Benchmarking Models for Brain Tumor Detection Using MRI Images

## Overview

This project presents a comprehensive benchmarking framework for brain tumor detection using MRI images. It evaluates multiple deep learning architectures — including classical convolutional neural networks (CNNs), hybrid quantum-classical models and quantum neural networks (QNNs) — to analyze their performance, robustness, and scalability. I used the same data for training/validating/testing of all models, to keep it fair and compare the performance of the models only. So what I was looking for was an answer to this question: we have CNNs ready at our disposal, QNNs are a new trend, is it worth it to go for hybrid QNN-CNN / pure QNN ? Do they give us any advantages ? Let's see !

The goal of this project is to provide a reproducible and extensible pipeline for comparing state-of-the-art models in medical image classification.

### Whole project in a glance:

* Step 1 : Trainin well known CNNs on top of ImageNet using transfer learning technique (VGG16/19, DenseNet121/201, etc.)
* Step 2 : Using the trained models in Step 1 for feature extraction then on top of that having quantum layers based on again transfer learning technique (PennyLane + JAX)
* Step 3 : All quantum approach (TBD)

---

## Objectives

* Benchmark widely-used CNN architectures for tumor detection
* Explore hybrid Quantum Neural Network (QNN) + CNN models
* Explore QNNs as a fully replacement of CNNs
* Evaluate models using robust metrics beyond accuracy
* Provide a reproducible and modular experimentation pipeline

---

## Models Evaluated

### Classical Models

* VGG16 / VGG19
* ResNet50V2
* DenseNet121 / DenseNet201
* EfficientNetB0
* MobileNetV2
* InceptionV3
* Xception

### Hybrid Models

* CNN feature extractor + Quantum layer (PennyLane + JAX)
* Variable number of qubits (2, 4, 6, 8, 12, 16) with depth 2

### QNN Models
* QNN feature extractor + Quantum layer (PennyLane + JAX)
* Variable number of qubits (4,6,8,12,16) and depths (1,2,3)

---

## Evaluation Metrics

Each model is evaluated using:

* Accuracy
* Precision
* Recall (Sensitivity)
* Specificity
* F1-score
* ROC-AUC
* Confusion Matrix

---

## Dataset

* [Brain MRI images for tumor classification](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
* [brain-tumor-classification-mri](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
* [Preprocessed IXI MRI](https://www.kaggle.com/datasets/hamedamin/preprocessed-oasis-and-epilepsy-and-ixi)
* Binary classification: **tumor / no tumor**
* Using the first 2 sources you will have around 5k yes cases and 600 no cases, to generalize and equalize the datasets I used IXI no cases only, at the end around 5k yes and 5k no cases were present in the training!
* I also used augmentation methods on images (cropping, rotating, etc.) 

> ⚠️ Due to dataset licensing and privacy constraints, the data is not included in this repository.

---

## Reproducibility

### 1. Clone the repository

```bash
git clone https://github.com/mousavi-hn/Benchmarking-Models-for-Brain-Tumor-Detection-Using-MRI-Images.git
cd Benchmarking-Models-for-Brain-Tumor-Detection-Using-MRI-Images
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare dataset

* Download dataset from the provided source
* Place images in:

```
data/MRI_Images/
    ├── yes/
    ├── no/
    └── IXI_no/
```

### 4. Run training

```bash
python scripts/train_hybrid.py
```

---

## Project Structure

```
project/
│
├── CNN/                 
├── hybrid CNN + QNN/    
├── QNN/                
├── results/             # Outputs (models, plots, reports)
```

### Sub-project Structure

```
sub-project/
│
├── scripts/        # Entry-point scripts
├── src/            # Core modules (data, models, training)
    ├── configs
    ├── data/
    ├── models/
    ├── train/
    ├── evaluate/

```

---

## Results

Results are automatically saved as:

* CSV summaries
* JSON reports
* Training plots (accuracy & loss curves)

Example output:

```
results/
├── saved_models/
├── plots/
├── reports/
└── hybrid_benchmark_summary.csv
```

---

## Discussion (most favourite part!)

Here I have shared the results in tables sorted by low to high false negatives!

The important note here is that, as these are vital classification models, then logically false negatives are the worst to happen (meaning there is a cancer, yet the model classifies as no cancer is present). So my endeavour was that to find the models, or a pipeline of models, to make the FNs as low as possible (ideally 0), yet as you may guess, this can not be ensured. 

Now the dilemma is this: should we go for adopting such technologies in critical/vital areas like this at all ? What happens if we have even 1 FN, a human life is endagnered, morally disastrous.

I have an idea, I will explain it shortly, first the tables!

### CNNs

| model_name     |   fn |   fp |   tn |   tp |   recall_sensitivity |   accuracy |   precision |   specificity |   f1_score |   roc_auc |   training_time_sec |
|:---------------|-----:|-----:|-----:|-----:|---------------------:|-----------:|------------:|--------------:|-----------:|----------:|--------------------:|
| VGG16          |    2 |    4 | 1046 |  927 |             0.997847 |   0.996968 |    0.995704 |      0.99619  |   0.996774 |  0.999958 |            33731.1  |
| DenseNet201    |    3 |    1 | 1049 |  926 |             0.996771 |   0.997979 |    0.998921 |      0.999048 |   0.997845 |  0.999875 |            21920.5  |
| ResNet50V2     |    4 |    3 | 1047 |  925 |             0.995694 |   0.996463 |    0.996767 |      0.997143 |   0.99623  |  0.999884 |            13843.9  |
| VGG19          |    4 |    6 | 1044 |  925 |             0.995694 |   0.994947 |    0.993555 |      0.994286 |   0.994624 |  0.999836 |            41627.5  |
| Xception       |    5 |    5 | 1045 |  924 |             0.994618 |   0.994947 |    0.994618 |      0.995238 |   0.994618 |  0.999722 |            18547.7  |
| EfficientNetB0 |    5 |    7 | 1043 |  924 |             0.994618 |   0.993936 |    0.992481 |      0.993333 |   0.993548 |  0.999531 |             7221.86 |
| InceptionV3    |    7 |    5 | 1045 |  922 |             0.992465 |   0.993936 |    0.994606 |      0.995238 |   0.993534 |  0.999843 |             8785.7  |
| MobileNetV2    |    7 |    2 | 1048 |  922 |             0.992465 |   0.995452 |    0.997835 |      0.998095 |   0.995143 |  0.999678 |             5072.94 |
| DenseNet121    |    8 |   11 | 1039 |  921 |             0.991389 |   0.990399 |    0.988197 |      0.989524 |   0.98979  |  0.999664 |            14070.2  |

### Hybrids

#### 2 qubit

| model_name     |   fn |   fp |   tn |   tp |   accuracy |   precision |   recall_sensitivity |   specificity |   f1_score |   roc_auc |   n_qubits |   q_depth |   training_time_sec |   balanced_accuracy |
|:---------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------------------:|--------------:|-----------:|----------:|-----------:|----------:|--------------------:|--------------------:|
| ResNet50V2     |    2 |    7 | 1043 |  927 |   0.995452 |    0.992505 |             0.997847 |      0.993333 |   0.995169 |  0.99892  |          2 |         2 |            10529.8  |            0.99559  |
| DenseNet201    |    4 |    4 | 1046 |  925 |   0.995958 |    0.995694 |             0.995694 |      0.99619  |   0.995694 |  0.999825 |          2 |         2 |            11183.9  |            0.995942 |
| EfficientNetB0 |    5 |    2 | 1048 |  924 |   0.996463 |    0.99784  |             0.994618 |      0.998095 |   0.996226 |  0.998892 |          2 |         2 |            14851.5  |            0.996357 |
| Xception       |    6 |    7 | 1043 |  923 |   0.993431 |    0.992473 |             0.993541 |      0.993333 |   0.993007 |  0.999584 |          2 |         2 |            38853.2  |            0.993437 |
| VGG19          |    6 |    6 | 1044 |  923 |   0.993936 |    0.993541 |             0.993541 |      0.994286 |   0.993541 |  0.996399 |          2 |         2 |            29576.4  |            0.993914 |
| InceptionV3    |    6 |    3 | 1047 |  923 |   0.995452 |    0.99676  |             0.993541 |      0.997143 |   0.995148 |  0.995694 |          2 |         2 |             5658.54 |            0.995342 |
| DenseNet121    |    7 |    4 | 1046 |  922 |   0.994442 |    0.99568  |             0.992465 |      0.99619  |   0.99407  |  0.99695  |          2 |         2 |             7049.97 |            0.994328 |
| VGG16          |    7 |    5 | 1045 |  922 |   0.993936 |    0.994606 |             0.992465 |      0.995238 |   0.993534 |  0.99693  |          2 |         2 |            27244.5  |            0.993852 |

#### 4 qubit

| model_name     |   fn |   fp |   tn |   tp |   accuracy |   precision |   recall_sensitivity |   specificity |   f1_score |   roc_auc |   n_qubits |   q_depth |   training_time_sec |   balanced_accuracy |
|:---------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------------------:|--------------:|-----------:|----------:|-----------:|----------:|--------------------:|--------------------:|
| EfficientNetB0 |    3 |    4 | 1046 |  926 |   0.996463 |    0.995699 |             0.996771 |      0.99619  |   0.996235 |  0.999721 |          4 |         2 |            14118.9  |            0.996481 |
| VGG16          |    3 |    3 | 1047 |  926 |   0.996968 |    0.996771 |             0.996771 |      0.997143 |   0.996771 |  0.99684  |          4 |         2 |            27213.4  |            0.996957 |
| MobileNetV2    |    4 |    4 | 1046 |  925 |   0.995958 |    0.995694 |             0.995694 |      0.99619  |   0.995694 |  0.999079 |          4 |         2 |             9869.03 |            0.995942 |
| DenseNet201    |    4 |    5 | 1045 |  925 |   0.995452 |    0.994624 |             0.995694 |      0.995238 |   0.995159 |  0.998978 |          4 |         2 |            13069.1  |            0.995466 |
| InceptionV3    |    5 |    9 | 1041 |  924 |   0.992926 |    0.990354 |             0.994618 |      0.991429 |   0.992481 |  0.992033 |          4 |         2 |             6186.18 |            0.993023 |
| VGG19          |    5 |    8 | 1042 |  924 |   0.993431 |    0.991416 |             0.994618 |      0.992381 |   0.993015 |  0.997398 |          4 |         2 |            23595.2  |            0.993499 |
| DenseNet121    |    6 |   12 | 1038 |  923 |   0.990904 |    0.987166 |             0.993541 |      0.988571 |   0.990343 |  0.999693 |          4 |         2 |             8801.23 |            0.991056 |
| ResNet50V2     |    7 |   10 | 1040 |  922 |   0.99141  |    0.98927  |             0.992465 |      0.990476 |   0.990865 |  0.996403 |          4 |         2 |             9232.38 |            0.991471 |
| Xception       |   10 |    2 | 1048 |  919 |   0.993936 |    0.997828 |             0.989236 |      0.998095 |   0.993514 |  0.999651 |          4 |         2 |            38830.9  |            0.993665 |

#### 6 qubit

| model_name     |   fn |   fp |   tn |   tp |   accuracy |   precision |   recall_sensitivity |   specificity |   f1_score |   roc_auc |   n_qubits |   q_depth |   training_time_sec |   balanced_accuracy |
|:---------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------------------:|--------------:|-----------:|----------:|-----------:|----------:|--------------------:|--------------------:|
| ResNet50V2     |    2 |    7 | 1043 |  927 |   0.995452 |    0.992505 |             0.997847 |      0.993333 |   0.995169 |  0.997034 |          6 |         2 |             8652.22 |            0.99559  |
| EfficientNetB0 |    3 |    5 | 1045 |  926 |   0.995958 |    0.994629 |             0.996771 |      0.995238 |   0.995699 |  0.998928 |          6 |         2 |            14906    |            0.996004 |
| VGG16          |    5 |    4 | 1046 |  924 |   0.995452 |    0.99569  |             0.994618 |      0.99619  |   0.995153 |  0.996273 |          6 |         2 |            27357.7  |            0.995404 |
| VGG19          |    5 |    5 | 1045 |  924 |   0.994947 |    0.994618 |             0.994618 |      0.995238 |   0.994618 |  0.994569 |          6 |         2 |            26943.4  |            0.994928 |
| DenseNet201    |    6 |    0 | 1050 |  923 |   0.996968 |    1        |             0.993541 |      1        |   0.99676  |  0.998943 |          6 |         2 |            11244.8  |            0.996771 |
| DenseNet121    |    6 |    5 | 1045 |  923 |   0.994442 |    0.994612 |             0.993541 |      0.995238 |   0.994076 |  0.997766 |          6 |         2 |             8926.21 |            0.99439  |
| Xception       |    6 |    2 | 1048 |  923 |   0.995958 |    0.997838 |             0.993541 |      0.998095 |   0.995685 |  0.995228 |          6 |         2 |            36845.6  |            0.995818 |
| MobileNetV2    |    7 |    6 | 1044 |  922 |   0.993431 |    0.993534 |             0.992465 |      0.994286 |   0.992999 |  0.99972  |          6 |         2 |             9333.81 |            0.993375 |
| InceptionV3    |    9 |    3 | 1047 |  920 |   0.993936 |    0.99675  |             0.990312 |      0.997143 |   0.993521 |  0.992083 |          6 |         2 |             6218.4  |            0.993728 |

#### 8 qubit

| model_name     |   fn |   fp |   tn |   tp |   accuracy |   precision |   recall_sensitivity |   specificity |   f1_score |   roc_auc |   n_qubits |   q_depth |   training_time_sec |   balanced_accuracy |
|:---------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------------------:|--------------:|-----------:|----------:|-----------:|----------:|--------------------:|--------------------:|
| ResNet50V2     |    2 |    9 | 1041 |  927 |   0.994442 |    0.990385 |             0.997847 |      0.991429 |   0.994102 |  0.999842 |          8 |         2 |             9336.92 |            0.994638 |
| EfficientNetB0 |    3 |    4 | 1046 |  926 |   0.996463 |    0.995699 |             0.996771 |      0.99619  |   0.996235 |  0.998993 |          8 |         2 |            14936.8  |            0.996481 |
| MobileNetV2    |    4 |    4 | 1046 |  925 |   0.995958 |    0.995694 |             0.995694 |      0.99619  |   0.995694 |  0.999679 |          8 |         2 |            10536.9  |            0.995942 |
| Xception       |    4 |    8 | 1042 |  925 |   0.993936 |    0.991426 |             0.995694 |      0.992381 |   0.993555 |  0.999489 |          8 |         2 |            36796.1  |            0.994038 |
| DenseNet201    |    5 |    3 | 1047 |  924 |   0.995958 |    0.996764 |             0.994618 |      0.997143 |   0.99569  |  0.999006 |          8 |         2 |            13050.6  |            0.99588  |
| DenseNet121    |    7 |    3 | 1047 |  922 |   0.994947 |    0.996757 |             0.992465 |      0.997143 |   0.994606 |  0.999751 |          8 |         2 |             8243.37 |            0.994804 |
| VGG16          |    7 |    3 | 1047 |  922 |   0.994947 |    0.996757 |             0.992465 |      0.997143 |   0.994606 |  0.997468 |          8 |         2 |            25179.9  |            0.994804 |
| VGG19          |    8 |    6 | 1044 |  921 |   0.992926 |    0.993528 |             0.991389 |      0.994286 |   0.992457 |  0.996723 |          8 |         2 |            29142    |            0.992837 |
| InceptionV3    |    8 |    3 | 1047 |  921 |   0.994442 |    0.996753 |             0.991389 |      0.997143 |   0.994064 |  0.995864 |          8 |         2 |             5734.22 |            0.994266 |

#### 12 qubit

| model_name     |   fn |   fp |   tn |   tp |   accuracy |   precision |   recall_sensitivity |   specificity |   f1_score |   roc_auc |   n_qubits |   q_depth |   training_time_sec |   balanced_accuracy |
|:---------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------------------:|--------------:|-----------:|----------:|-----------:|----------:|--------------------:|--------------------:|
| ResNet50V2     |    1 |    6 | 1044 |  928 |   0.996463 |    0.993576 |             0.998924 |      0.994286 |   0.996243 |  0.996338 |         12 |         2 |            11596.3  |            0.996605 |
| MobileNetV2    |    3 |   14 | 1036 |  926 |   0.99141  |    0.985106 |             0.996771 |      0.986667 |   0.990904 |  0.999552 |         12 |         2 |            10818.8  |            0.991719 |
| EfficientNetB0 |    5 |    3 | 1047 |  924 |   0.995958 |    0.996764 |             0.994618 |      0.997143 |   0.99569  |  0.999799 |         12 |         2 |            13400.1  |            0.99588  |
| Xception       |    5 |    7 | 1043 |  924 |   0.993936 |    0.992481 |             0.994618 |      0.993333 |   0.993548 |  0.999799 |         12 |         2 |            36951.5  |            0.993976 |
| VGG16          |    5 |    1 | 1049 |  924 |   0.996968 |    0.998919 |             0.994618 |      0.999048 |   0.996764 |  0.999039 |         12 |         2 |            25131.6  |            0.996833 |
| DenseNet201    |    6 |    1 | 1049 |  923 |   0.996463 |    0.998918 |             0.993541 |      0.999048 |   0.996222 |  0.999625 |         12 |         2 |            11329.6  |            0.996295 |
| DenseNet121    |    7 |    5 | 1045 |  922 |   0.993936 |    0.994606 |             0.992465 |      0.995238 |   0.993534 |  0.99977  |         12 |         2 |             9102.69 |            0.993852 |
| VGG19          |    7 |    6 | 1044 |  922 |   0.993431 |    0.993534 |             0.992465 |      0.994286 |   0.992999 |  0.991148 |         12 |         2 |            27128.9  |            0.993375 |
| InceptionV3    |   10 |    3 | 1047 |  919 |   0.993431 |    0.996746 |             0.989236 |      0.997143 |   0.992977 |  0.996955 |         12 |         2 |             5936.41 |            0.993189 |

#### 16 qubit

TBD!

### QNNs

---

## Key Contributions

* Unified benchmarking of multiple CNN architectures
* Integration of hybrid quantum-classical models
* Modular and reproducible ML pipeline
* Evaluation using medically relevant metrics

---

## Future Work

* Tumor segmentation (e.g., U-Net architectures)
* Model explainability (Grad-CAM, saliency maps)
* Hyperparameter optimization
* Clinical dataset validation

---

## References

* Deep Learning for Medical Image Analysis
* Quantum Machine Learning frameworks (PennyLane, JAX)
* Transfer learning in medical imaging

---

## Acknowledgments

This project is developed as part of ongoing research and study in machine learning and medical imaging.

---

## Contact

For questions or collaboration:

* GitHub: https://github.com/mousavi-hn
* Email: mousavi.hn@gmail.com

