import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import RMSprop
from from_root import from_root

root = from_root()

train_dir = os.path.join(from_root(), 'train')
test_dir = os.path.join(from_root(), 'test')
validation_dir = os.path.join(from_root(), 'val')

np.random.seed(45)
tf.random.set_seed(45)

# generating batches of tensor image data
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=10,
                                   # width_shift_range=0.2,
                                   # height_shift_range=0.2,
                                   # shear_range=0.3,
                                   # zoom_range=0.4,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=81,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=90,
    class_mode='binary')

conv_base = VGG19(weights='imagenet',
                  include_top=False,
                  input_shape=(128, 128, 3))

conv_base.summary()

conv_base.trainable = False

fcnn_1 = keras.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(300, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

fcnn_1.summary()

fcnn_1.compile(loss='binary_crossentropy',
               optimizer=RMSprop(learning_rate=2e-5),
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint("Mask_or_Nomask_model_custom.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(min_delta=0.001,
                                                  patience=3,
                                                  restore_best_weights=True)

history = fcnn_1.fit(train_generator,
                     steps_per_epoch=100,
                     epochs=20,
                     validation_data=validation_generator,
                     validation_steps=30,
                     callbacks=[checkpoint_cb, early_stopping_cb])

pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary')

fcnn_1.evaluate(test_generator, steps=50)
