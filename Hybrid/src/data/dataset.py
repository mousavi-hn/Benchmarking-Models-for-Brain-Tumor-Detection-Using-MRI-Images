"""
Collect MRI image paths and associated dataset metadata.

The MRI dataset contains images from the IXI dataset and additional non-IXI
sources. Each discovered image is assigned a binary cancer label and source
identifier. IXI images additionally retain their subject identifier so that
subject-level splitting can be performed without cross-partition leakage.
"""
import os
from pathlib import Path

import pandas as pd

from src.configs import VALID_EXTENSIONS

def collect_image_paths(dataset_dir):
    """
    Collect MRI image paths, labels, source information, and subject metadata.

    Images under the ``no`` and ``yes`` directories are treated as non-IXI
    samples and mapped to labels 0 and 1, respectively. Images under
    ``IXI_no`` are negative samples from the IXI dataset and receive a
    subject identifier derived from their containing directory.

    Args:
        dataset_dir: Root directory containing the MRI dataset.

    Returns:
        pandas.DataFrame: One row per discovered image with filepath, label,
        class name, source, and subject identifier.

    Raises:
        FileNotFoundError: If an expected class or IXI directory is missing.
        ValueError: If no supported image files are found.
    """
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