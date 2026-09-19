"""Configuration, paths, training hyperparameters, and experiment settings for the hybrid pipeline."""
import keras
import os
from pathlib import Path

def find_absolute_path_to_images():
    """Resolve the project image directory.
    
    Returns:
        pathlib.Path: Absolute path to the project ``images`` directory.
    """
    config_file = Path(__file__).resolve()
    proj_root = config_file.parents[2]
    images_dir = proj_root / "images"

    return images_dir

def find_absolute_path_to_results_folder():
    """Resolve the experiment results directory.
    
    Returns:
        pathlib.Path: Absolute path to the configured results directory.
    """
    config_file = Path(__file__).resolve()
    proj_root = config_file.parents[2]
    results_dir = proj_root / "results/CT_hybrid_benchmark_results"

    return results_dir

def find_absolute_path_to_classical_models():
    """Resolve the directory containing trained classical models.
    
    Returns:
        pathlib.Path: Absolute path to the classical model directory.
    """
    config_file = Path(__file__).resolve()
    proj_root = config_file.parents[2]
    models_dir = proj_root / "results/CT_cnn_benchmark_results/saved_models"

# PATHS AND SETTINGS
DATASET_DIR = find_absolute_path_to_images()
OUTPUT_DIR = find_absolute_path_to_results_folder()
SPLIT_DIR = os.path.join(OUTPUT_DIR, "splits")
CLASSICAL_MODEL_DIR = find_absolute_path_to_classical_models()
MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

EPOCHS_HEAD = 8
EPOCHS_FINE = 5

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5

QUANTUM_QUBITS = [2, 4, 6, 8, 12, 16]
Q_DEPTH = 2

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

MODEL_CONFIGS = {
    "VGG16": {
        "preprocess": keras.applications.vgg16.preprocess_input
    },
    "VGG19": {
        "preprocess": keras.applications.vgg19.preprocess_input
    },
    "ResNet50V2": {
        "preprocess": keras.applications.resnet_v2.preprocess_input
    },
    "DenseNet121": {
        "preprocess": keras.applications.densenet.preprocess_input
    },
    "DenseNet201": {
        "preprocess": keras.applications.densenet.preprocess_input
    },
    "EfficientNetB0": {
        "preprocess": keras.applications.efficientnet.preprocess_input
    },
    "MobileNetV2": {
        "preprocess": keras.applications.mobilenet_v2.preprocess_input
    },
    "InceptionV3": {
        "preprocess": keras.applications.inception_v3.preprocess_input
    },
    "Xception": {
        "preprocess": keras.applications.xception.preprocess_input
    }
}

MODEL_NAMES = list(MODEL_CONFIGS.keys())