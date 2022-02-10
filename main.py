import numpy as np
import tensorflow as tf
from src.utils import generator, generator_obj
from src.model import build_model
from validation import validate_result
import os
from tensorflow.keras.optimizers import RMSprop
from from_root import from_root
from src.checkpointing import saveCheckpoint


# Setting the environment Seed
def set_seed(seed=45):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Resolving root issues
root = from_root()
train_dir = os.path.join(from_root(), 'train')
test_dir = os.path.join(from_root(), 'test')
validation_dir = os.path.join(from_root(), 'val')

# Creating Data generator
train_datagen, test_datagen = generator_obj()

# Image Generation
train_generator, validation_generator = generator(train_datagen, test_datagen, train_dir, validation_dir)

# Model return
model = build_model()

# model compile
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=2e-5),
              metrics=['accuracy'])

checkpoint_cb, early_stopping_cb = saveCheckpoint()

history = model.fit(train_generator,
                    steps_per_epoch=10,
                    epochs=1,
                    validation_data=validation_generator,
                    validation_steps=5,
                    callbacks=[checkpoint_cb, early_stopping_cb])

validation = validate_result(test_datagen, test_dir, model)

if __name__ == "__main__":
    print(history)
