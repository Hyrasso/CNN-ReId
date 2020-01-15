def get_model(n_classes: int):

    input_tensor = Input(shape=(224, 224, 3))  # channels_last

    # create the base pre-trained model
    base_model = ResNet50(input_tensor=input_tensor,weights='imagenet',include_top=False)

    for layer in base_model.layers:
        layer.trainable=False

    x = base_model.output
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dense(n_classes, activation='sigmoid')(x)

    updatedModel = Model(base_model.input, x)

    return updatedModel

# loss binary crossentropy?
# age not binary
