from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers
from tensorflow import keras

def build_model():
    conv_base = VGG19(weights='imagenet',
                      include_top=False,
                      input_shape=(128, 128, 3))

    conv_base.trainable = False

    model = keras.Sequential([
        conv_base,
        layers.Flatten(),
        layers.Dense(300, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    print(model.summary())
    return model
