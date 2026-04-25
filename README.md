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

Here I have shared the summary results, top 10 models, in a table sorted by low to high false negatives!

The important note here is this: these are vital classification models, logically false negatives are the worst to happen (patient has cancer, yet the model classifies as no cancer is present). So my endeavour was that to find the models, or a pipeline of models, to make the FNs as low as possible (ideally 0), yet as you may guess, this can not be ensured. 

The dilemma is this: should we go for adopting such technologies in critical/vital areas like this at all ? What happens if we have even 1 FN, a human life is endagnered, morally disastrous.

To answer the dilemma, we can use these pipelines to priorotize the cases, those reported with cancer will be dealt with, yet those reported as no cancer will be double checked by the specialist as well, so using engineering techniques not to replace the doctor, but to help them organize the work.

| model_name     |   fn |   fp |   tn |   tp |   accuracy |   precision |   recall_sensitivity |   specificity |   f1_score |   roc_auc |   n_qubits |   q_depth |   training_time_sec |          model type |
|:---------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------------------:|--------------:|-----------:|----------:|-----------:|----------:|--------------------:|--------------------:|
| ResNet50V2     |    1 |    6 | 1044 |  928 |   0.996463 |    0.993576 |             0.998924 |      0.994286 |   0.996243 |  0.996338 |         12 |         2 |            11596.3  |              hybrid |
| ResNet50V2     |    2 |    7 | 1043 |  927 |   0.995452 |    0.992505 |             0.997847 |      0.993333 |   0.995169 |  0.99892  |          2 |         2 |            10529.8  |              hybrid |
| ResNet50V2     |    2 |    7 | 1043 |  927 |   0.995452 |    0.992505 |             0.997847 |      0.993333 |   0.995169 |  0.997034 |          6 |         2 |             8652.22 |              hybrid |
| ResNet50V2     |    2 |    9 | 1041 |  927 |   0.994442 |    0.990385 |             0.997847 |      0.991429 |   0.994102 |  0.999842 |          8 |         2 |             9336.92 |              hybrid |
| VGG16          |    2 |    4 | 1046 |  927 |   0.996968 |    0.995704 |             0.997847 |      0.99619  |   0.996774 |  0.999958 |       None |      None |             33731.1 |       classical CNN |
| EfficientNetB0 |    3 |    4 | 1046 |  926 |   0.996463 |    0.995699 |             0.996771 |      0.99619  |   0.996235 |  0.999721 |          4 |         2 |             14118.9 |              hybrid |
| EfficientNetB0 |    3 |    5 | 1045 |  926 |   0.995958 |    0.994629 |             0.996771 |      0.995238 |   0.995699 |  0.998928 |          6 |         2 |             14906   |              hybrid |
| EfficientNetB0 |    3 |    4 | 1046 |  926 |   0.996463 |    0.995699 |             0.996771 |      0.99619  |   0.996235 |  0.998993 |          8 |         2 |             14936.8 |              hybrid |
| DenseNet201    |    3 |    1 | 1049 |  926 |   0.997979 |    0.998921 |             0.996771 |      0.999048 |   0.997845 |  0.999875 |       None |      None |             21920.5 |       classical CNN |
| VGG16          |    3 |    3 | 1047 |  926 |   0.996968 |    0.996771 |             0.996771 |      0.997143 |   0.996771 |  0.99684  |          4 |         2 |             27213.4 |              hybrid |

(please notice that although the model names are repetative, the number of qubits/depth is different, in nature, they are different models!)

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

