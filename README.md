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
| VGG16          |    2 |    4 | 1046 |  927 |   0.996968 |    0.995704 |             0.997847 |      0.99619  |   0.996774 |  0.999958 |       None |      None |             33731.1 |       classical CNN |
| DenseNet201    |    3 |    1 | 1049 |  926 |   0.997979 |    0.998921 |             0.996771 |      0.999048 |   0.997845 |  0.999875 |       None |      None |             21920.5 |       classical CNN |
| EfficientNetB0 |    3 |    4 | 1046 |  926 |   0.996463 |    0.995699 |             0.996771 |      0.99619  |   0.996235 |  0.999721 |          4 |         2 |             14118.9 |              hybrid |
| MobileNetV2    |    3 |   14 | 1036 |  926 |   0.99141  |    0.985106 |             0.996771 |      0.986667 |   0.990904 |  0.999552 |         12 |         2 |            10818.8  |              hybrid |
| VGG19          |    4 |    6 | 1044 |  925 |   0.994947 |    0.993555 |             0.995694 |      0.994286 |   0.994624 |  0.999836 |       None |      None |             41627.5 |       classical CNN |
| Xception       |    4 |    8 | 1042 |  925 |   0.993936 |    0.991426 |             0.995694 |      0.992381 |   0.993555 |  0.999489 |          8 |         2 |            36796.1  |              hybrid |
| InceptionV3    |    5 |    9 | 1041 |  924 |   0.992926 |    0.990354 |             0.994618 |      0.991429 |   0.992481 |  0.992033 |          4 |         2 |             6186.18 |              hybrid |
| DenseNet121    |    6 |    5 | 1045 |  923 |   0.994442 |    0.994612 |             0.993541 |      0.995238 |   0.994076 |  0.997766 |          6 |         2 |             8926.21 |              hybrid |
| QNN            |    (TBD!)                                                                                                                                                           


### Some important points about the table:
* I compared each model with its hybrid variants, I chose the one which outperforms in terms of lower false negatives
* Among 9 different models, 6 hybrids were chosen and 3 classical, a signal that emphasizes the future of quantum ML
* I have provided the full tables with all the variants in TABLES.md, you find it in the main page of repository
* The interesting part is that I have checked (2,4,6,8,12,16) qubit versions, higher qubits do not mean we are going to have a better model, e.g. EfficientNetB0 with a 4 qubit head excels a 16 qubit head
* The ultimate goal of the benchmarking is to provide a backbone to an ensemble decision making software, checking the presence of different tumors in the given images, so I kept the list heterogeneous to generalize better
* More discussion and visualization in the notebooks (will be available soon!)

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

## Pretrained models can be found here:
https://drive.google.com/drive/folders/1aWNdRHJqgsHlvnzahH66naxPWpQ-hzuQ?usp=sharing

* Due to high volume of the trained models, I have shared them in a google drive link! Please feel free to have a look and use the models for your work, in that case I would be glad if you please let me know about it, yet there are no licensing restrictions here, all is my independent work!

---

## Contact

For questions or collaboration:

* GitHub: https://github.com/mousavi-hn
* Email: mousavi.hn@gmail.com

