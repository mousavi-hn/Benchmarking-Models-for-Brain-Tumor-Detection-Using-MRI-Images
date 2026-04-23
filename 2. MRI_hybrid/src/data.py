import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import random
from PIL import Image

import numpy as np
import pandas as pd

import keras

from sklearn.model_selection import train_test_split

# =========================================================
# 1. REPRODUCIBILITY
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

# =========================================================
# 2. PATHS AND SETTINGS
# =========================================================
DATASET_DIR = "../data/MRI_Images"          # yes / no / IXI_no
CLASSICAL_MODEL_DIR = "../results/MRI_cnn_benchmark_results/saved_models"
OUTPUT_DIR = "../results/MRI_hybrid_benchmark_results"
SPLIT_DIR = os.path.join(OUTPUT_DIR, "splits")
MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

for folder in [OUTPUT_DIR, SPLIT_DIR, MODEL_DIR, PLOT_DIR, REPORT_DIR]:
    os.makedirs(folder, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 8                    # safer for hybrid experiments
EPOCHS_HEAD = 8
EPOCHS_FINE = 5
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5

QUANTUM_QUBITS = [2, 4, 6, 8, 12, 16]
Q_DEPTH = 2

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Optional: start with only one model while debugging
DEBUG_SINGLE_MODEL = "MobileNetV2"  # e.g. "DenseNet121"
DEBUG_SINGLE_QUBITS = [2] # e.g. [2, 4]

# =========================================================
# 4. DATA COLLECTION
# =========================================================
def collect_image_paths(dataset_dir):
    records = []

    class_map = {
        "no": 0,
        "yes": 1,
    }

    for class_name, label in class_map.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Folder not found: {class_dir}")

        for root, _, files in os.walk(class_dir):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in VALID_EXTENSIONS:
                    records.append({
                        "filepath": os.path.join(root, file),
                        "label": label,
                        "class_name": class_name,
                        "source": "non_IXI",
                        "subject_id": None
                    })

    ixi_dir = os.path.join(dataset_dir, "IXI_no")
    if not os.path.exists(ixi_dir):
        raise FileNotFoundError(f"Folder not found: {ixi_dir}")

    for root, _, files in os.walk(ixi_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in VALID_EXTENSIONS:
                subject_id = Path(root).name
                records.append({
                    "filepath": os.path.join(root, file),
                    "label": 0,
                    "class_name": "no",
                    "source": "IXI",
                    "subject_id": subject_id
                })

    full_df = pd.DataFrame(records)
    if full_df.empty:
        raise ValueError("No images found. Check dataset path and file extensions.")

    return full_df

# =========================================================
# 5. STRATIFIED SPLIT (70 / 15 / 15)
# =========================================================
def make_splits(full_df):
    ixi_df = full_df[full_df["source"] == "IXI"].copy()
    other_df = full_df[full_df["source"] != "IXI"].copy()

    other_train, other_temp = train_test_split(
        other_df,
        test_size=0.30,
        stratify=other_df["label"],
        random_state=SEED
    )

    other_val, other_test = train_test_split(
        other_temp,
        test_size=0.50,
        stratify=other_temp["label"],
        random_state=SEED
    )

    ixi_subjects = ixi_df[["subject_id"]].drop_duplicates()

    ixi_train_subjects, ixi_temp_subjects = train_test_split(
        ixi_subjects,
        test_size=0.30,
        random_state=SEED
    )

    ixi_val_subjects, ixi_test_subjects = train_test_split(
        ixi_temp_subjects,
        test_size=0.50,
        random_state=SEED
    )

    ixi_train = ixi_df[ixi_df["subject_id"].isin(ixi_train_subjects["subject_id"])]
    ixi_val = ixi_df[ixi_df["subject_id"].isin(ixi_val_subjects["subject_id"])]
    ixi_test = ixi_df[ixi_df["subject_id"].isin(ixi_test_subjects["subject_id"])]

    train_df = pd.concat([other_train, ixi_train], ignore_index=True)
    val_df = pd.concat([other_val, ixi_val], ignore_index=True)
    test_df = pd.concat([other_test, ixi_test], ignore_index=True)

    train_df.to_csv(os.path.join(SPLIT_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(SPLIT_DIR, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(SPLIT_DIR, "test_split.csv"), index=False)

    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return train_df, val_df, test_df

# =========================================================
# 6. IMAGE LOADER / DATASET
# =========================================================
def read_image(path, target_size):
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32)
    return arr

class MRISequence(keras.utils.Sequence):
    def __init__(self, df, preprocess_func, batch_size=8, target_size=(224, 224), shuffle=False):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.preprocess_func = preprocess_func
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indices]

        images = []
        labels = []

        for _, row in batch_df.iterrows():
            img = read_image(row["filepath"], self.target_size)
            img = self.preprocess_func(img)
            images.append(img)
            labels.append(float(row["label"]))

        x = np.asarray(images, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(SEED)
            rng.shuffle(self.indices)

def make_generators(preprocess_func, train_df, val_df, test_df):
    train_seq = MRISequence(
        train_df, preprocess_func, batch_size=BATCH_SIZE,
        target_size=IMG_SIZE, shuffle=True
    )
    val_seq = MRISequence(
        val_df, preprocess_func, batch_size=BATCH_SIZE,
        target_size=IMG_SIZE, shuffle=False
    )
    test_seq = MRISequence(
        test_df, preprocess_func, batch_size=BATCH_SIZE,
        target_size=IMG_SIZE, shuffle=False
    )
    return train_seq, val_seq, test_seq