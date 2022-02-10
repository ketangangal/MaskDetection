def validate_result(test_datagen=None, test_dir=None, model=None):
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=20,
        class_mode='binary')

    print(model.evaluate(test_generator, steps=50))
    return True
