import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

#  MODEL LIST
MODEL_CONFIGS = {
    "VGG16": {
        "builder": tf.keras.applications.VGG16,
        "preprocess": tf.keras.applications.vgg16.preprocess_input
    },
    "VGG19": {
        "builder": tf.keras.applications.VGG19,
        "preprocess": tf.keras.applications.vgg19.preprocess_input
    },
    "ResNet50V2": {
        "builder": tf.keras.applications.ResNet50V2,
        "preprocess": tf.keras.applications.resnet_v2.preprocess_input
    },
    "DenseNet121": {
        "builder": tf.keras.applications.DenseNet121,
        "preprocess": tf.keras.applications.densenet.preprocess_input
    },
    "DenseNet201": {
        "builder": tf.keras.applications.DenseNet201,
        "preprocess": tf.keras.applications.densenet.preprocess_input
    },
    "EfficientNetB0": {
        "builder": tf.keras.applications.EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input
    },
    "MobileNetV2": {
        "builder": tf.keras.applications.MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input
    },
    "InceptionV3": {
        "builder": tf.keras.applications.InceptionV3,
        "preprocess": tf.keras.applications.inception_v3.preprocess_input
    },
    "Xception": {
        "builder": tf.keras.applications.Xception,
        "preprocess": tf.keras.applications.xception.preprocess_input
    }
}

MODEL_NAMES = list(MODEL_CONFIGS.keys())

# BUILD MODEL
def build_transfer_model(model_name, input_shape=(224, 224, 3), dropout_rate=0.3):
    config = MODEL_CONFIGS[model_name]
    base_builder = config["builder"]

    base_model = base_builder(
        include_top=False,
        weights="imagenet",
        input_tensor=Input(shape=input_shape)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model