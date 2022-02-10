from tensorflow import keras


def saveCheckpoint(path=None):

    checkpoint_cb = keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True,)
    early_stopping_cb = keras.callbacks.EarlyStopping(min_delta=0.001,
                                                      patience=3,
                                                      restore_best_weights=True)
    return checkpoint_cb, early_stopping_cb

