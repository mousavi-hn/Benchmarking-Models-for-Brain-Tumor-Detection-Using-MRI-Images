import os
import random
import warnings

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")

# REPRODUCIBILITY
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# PATHS AND SETTINGS
DATASET_DIR = "../data/MRI"
OUTPUT_DIR = "../results/MRI_cnn_benchmark_results"
SPLIT_DIR = os.path.join(OUTPUT_DIR, "splits")
MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# For fairness, I have used same input size for all models.
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 8         # training classifier head first
EPOCHS_FINE = 7         # then fine-tuning upper layers
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# LOAD FILE PATHS
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

full_df = collect_image_paths(DATASET_DIR)
print("Total images:", len(full_df))
print(full_df["class_name"].value_counts())

# STRATIFIED SPLIT (70 / 15 / 15)
# As datasets come from kaggle, they may have duplicates, so on non-IXI images I did a cleaning to find and remove duplicates
# then I added IXI images ( they come with patient ID so there is no duplicate among them )
# Separate IXI and non-IXI
ixi_df = full_df[full_df["source"] == "IXI"].copy()
other_df = full_df[full_df["source"] != "IXI"].copy()

# Split non-IXI normally by label ( labels are no and yes, no says there is no cancer and yes the opposite )
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

# Split IXI by subject_id
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

# Merge splits
train_df = pd.concat([other_train, ixi_train], ignore_index=True)
val_df = pd.concat([other_val, ixi_val], ignore_index=True)
test_df = pd.concat([other_test, ixi_test], ignore_index=True)

# saving CSVs
train_df.to_csv(os.path.join(SPLIT_DIR, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(SPLIT_DIR, "val_split.csv"), index=False)
test_df.to_csv(os.path.join(SPLIT_DIR, "test_split.csv"), index=False)

# shuffle rows
train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Check class balance
print("Train label counts:\n", train_df["class_name"].value_counts())
print("Val label counts:\n", val_df["class_name"].value_counts())
print("Test label counts:\n", test_df["class_name"].value_counts())

# Check IXI subject leakage
print("Train IXI subjects:", ixi_train["subject_id"].nunique())
print("Val IXI subjects:", ixi_val["subject_id"].nunique())
print("Test IXI subjects:", ixi_test["subject_id"].nunique())

# GENERATORS
def make_generators(preprocess_func, train_df, val_df, test_df, target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.10,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="class_name",
        target_size=target_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=True,
        seed=SEED
    )

    val_gen = test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="class_name",
        target_size=target_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col="class_name",
        target_size=target_size,
        color_mode="rgb",
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen, test_gen