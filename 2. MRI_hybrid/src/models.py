import os

import keras
from keras import layers
from keras import Model

import pennylane as qml
import jax.numpy as jnp
import jax

from data import CLASSICAL_MODEL_DIR, IMG_SIZE

# MODEL PREPROCESSING MAP
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

# =========================================================
# 7. QUANTUM LAYER
# =========================================================
class QuantumLayer(layers.Layer):
    def __init__(self, n_qubits, q_depth=2, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="jax")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def build(self, input_shape):
        self.q_weights = self.add_weight(
            name="q_weights",
            shape=(self.q_depth, self.n_qubits, 3),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        inputs = jnp.asarray(inputs)
        weights = jnp.asarray(self.q_weights)

        def single_forward(x):
            return self.circuit(x, weights)

        outputs = jax.vmap(single_forward)(inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "q_depth": self.q_depth,
        })
        return config

# =========================================================
# 8. CLASSICAL MODEL LOADING
# =========================================================
def find_classical_model_path(model_name):
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

# =========================================================
# 9. HYBRID MODEL
# =========================================================
def build_hybrid_model(model_path, n_qubits, q_depth=2, freeze_backbone=True):
    feature_extractor = load_feature_extractor(model_path)
    feature_extractor.trainable = not freeze_backbone

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    x = feature_extractor(inputs, training=False)

    # Compress classical features to match qubit count
    x = layers.Dense(64, activation="relu", name="pre_quantum_dense")(x)
    x = layers.Dropout(0.2, name="pre_quantum_dropout")(x)
    x = layers.Dense(n_qubits, activation="tanh", name="quantum_input_projection")(x)

    x = QuantumLayer(n_qubits=n_qubits, q_depth=q_depth, name=f"quantum_{n_qubits}q")(x)

    x = layers.Dense(16, activation="relu", name="post_quantum_dense")(x)
    x = layers.Dropout(0.2, name="post_quantum_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="hybrid_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"hybrid_{n_qubits}q")
    return model, feature_extractor

def unfreeze_top_fraction(feature_extractor, fraction=0.30):
    feature_extractor.trainable = True
    fine_tune_at = int(len(feature_extractor.layers) * (1 - fraction))

    for layer in feature_extractor.layers[:fine_tune_at]:
        layer.trainable = False

    for layer in feature_extractor.layers[fine_tune_at:]:
        layer.trainable = True

    # Keep normalization layers frozen for more stable fine-tuning
    for layer in feature_extractor.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False