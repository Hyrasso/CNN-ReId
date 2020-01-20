# import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import Input, Model
from keras.layers import Dense, Lambda, MaxPooling2D, Flatten
import numpy as np
from utils.marketsequence import MarketSequence
from utils.preprocess import resize

def resize_preprocess(array):
    # TODO: resize to 224, 224, 3, from 128, 64, 3
    # array = resize(array, (224, 224, 3))

    # r = K.resize_images(img, 224 / 128, 224 / 64, data_format="channels_last", interpolation="bilinear")
    # assert array[0].shape == (224, 224, 3)
    # array = resize_images(array, 224 / array.shape[1], 224 / array.shape[2], "channels_last", interpolation="bilinear")
    # preprocess
    array = preprocess_input(array)

    return array

def preprocess_labels(array):
    array = array[:, 1:] # remove age
    array = array - 1 # 1-2 to binary
    return array

def get_model(n_classes: int):

    input_tensor = Input(shape=(128, 64, 3))  # channels_last

    # create the base pre-trained model
    base_model = ResNet50(input_tensor=input_tensor, input_shape=(128, 64, 3), pooling="avg", weights='imagenet',include_top=False)

    for layer in base_model.layers:
        layer.trainable=False

    x = base_model.output
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
test_ms = MarketSequence("market.h5", 1024, train=False, preprocess=(resize_preprocess, preprocess_labels))
hist = model.fit_generator(train_ms, epochs=5)
# model.fit(*train_ms[0], batch_size=16)
# print(model.evaluate(*test_ms[0]))
print(hist.history)
model.save("model_attr.h5")
