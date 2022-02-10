from tensorflow.keras.preprocessing.image import ImageDataGenerator


def generator_obj():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=10,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    return train_datagen, test_datagen


def generator(train_datagen, test_datagen, train_dir, validation_dir):
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
    return train_generator, validation_generator
