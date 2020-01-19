from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from keras import Input, Model
from keras.layers import Dense, Lambda, MaxPooling2D, Flatten
from utils.marketsequence import MarketSequence
from utils.preprocess import resize

def resize_preprocess(array):
    # TODO: resize to 224, 224, 3, from 128, 64, 3
    res = resize(array, (224, 224, 3))

    # r = K.resize_images(img, 224 / 128, 224 / 64, data_format="channels_last", interpolation="bilinear")
    # assert array[0].shape == (224, 224, 3)
    # array = resize_images(array, 224 / array.shape[1], 224 / array.shape[2], "channels_last", interpolation="bilinear")
    # preprocess
    res = preprocess_input(res)

    return res

def preprocess_labels(array):
    array = array[:, 1:] # remove age
    array = array - 1 # 1-2 to binary
    return array

def get_model(n_classes: int):

    input_tensor = Input(shape=(224, 224, 3))  # channels_last

    # create the base pre-trained model
    base_model = ResNet50(input_tensor=input_tensor,weights='imagenet',include_top=False)

    for layer in base_model.layers:
        layer.trainable=False

    x = base_model.output
    x = MaxPooling2D((3,3), data_format='channels_last')(x)
    x = Flatten()(x)
    x = Dense(n_classes, activation='sigmoid')(x)
    # Fully connected -> 512
    # batch norm
    # dropout
    # relu
    updatedModel = Model(base_model.input, x)

    return updatedModel

# loss binary crossentropy?
# age not binary


# def i(a):return a

train_ms = MarketSequence("market.h5", 32, preprocess=(resize_preprocess, preprocess_labels))

model = get_model(n_classes=26)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# model.summary()
model.fit_generator(train_ms, epochs=1)

model.evaluate_generator(MarketSequence("market.h5", 32, train=False, preprocess=(resize_preprocess, preprocess_labels)))
