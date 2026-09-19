"""Locate trained classical CNNs and expose their penultimate representations as feature extractors."""
import os

import keras
from keras import Model

from src.configs import CLASSICAL_MODEL_DIR

def find_classical_model_path(model_name):
    """Find the saved Keras model for a classical CNN backbone.
    
    Args:
        model_name: Backbone name used to construct candidate filenames.
    
    Returns:
        str: Path to the first matching saved model.
    
    Raises:
        FileNotFoundError: If no matching ``.keras`` model exists.
    """
    candidates = [
        os.path.join(CLASSICAL_MODEL_DIR, f"{model_name}.keras"),
        os.path.join(CLASSICAL_MODEL_DIR, f"{model_name}_best.keras"),
        os.path.join(CLASSICAL_MODEL_DIR, f"{model_name.lower()}.keras"),
        os.path.join(CLASSICAL_MODEL_DIR, f"{model_name.lower()}_best.keras"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No .keras model found for {model_name} in {CLASSICAL_MODEL_DIR}"
    )

def load_feature_extractor(model_path):
    """Load a classical model and expose its penultimate layer as features.
    
    Args:
        model_path: Path to a saved Keras classical model.
    
    Returns:
        keras.Model: Model mapping input images to the penultimate representation.
    
    Raises:
        ValueError: If the loaded model does not contain enough layers.
    """
    loaded_model = keras.models.load_model(model_path, compile=False)

    if len(loaded_model.layers) < 2:
        raise ValueError(f"Model at {model_path} does not have enough layers.")

    # We remove the last sigmoid layer and keep the penultimate representation
    feature_extractor = Model(
        inputs=loaded_model.input,
        outputs=loaded_model.layers[-2].output,
        name=f"{loaded_model.name}_feature_extractor"
    )
    return feature_extractor