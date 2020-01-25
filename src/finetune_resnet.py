# import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import Input, Model
from keras.metrics import binary_accuracy, categorical_accuracy
from keras.layers import Dense, Lambda, MaxPooling2D, Flatten, Concatenate, BatchNormalization, Dropout, Multiply
import keras.backend as K
import numpy as np
from utils.marketsequence import MarketSequence
from utils.preprocess import resize
from typing import Tuple

def preprocess_images(array):
    # TODO: resize to 224, 224, 3, from 128, 64, 3
    # array = resize(array, (224, 224, 3))

    # r = K.resize_images(img, 224 / 128, 224 / 64, data_format="channels_last", interpolation="bilinear")
    # assert array[0].shape == (224, 224, 3)
    # array = resize_images(array, 224 / array.shape[1], 224 / array.shape[2], "channels_last", interpolation="bilinear")
    # preprocess
    array = preprocess_input(array)

    return array

def preprocess_labels(array):
    # 1, 2 -> 0 (young)
    # 3, 4 -> adults
    array[:, 0] = np.array(array[:, 0] > 2, dtype=float)
    array[:, 1:] = array[:, 1:] - 1 # 1-2 to binary
    return array

def get_models(n_person: int=1501, n_attributes: int=27) -> Tuple[Model, Model, Model]:
    """ Returns attribute prediction model, id prediction model and embedding model 
    
    embedding model is the id prediction model without the last classification layer
    """

    input_tensor = Input(shape=(128, 64, 3))  # channels_last

    # create the base pre-trained model
    base_model = ResNet50(input_shape=(128, 64, 3), pooling="avg", weights='imagenet',include_top=False)(input_tensor)

    attribute_layer = Dense(n_attributes, activation="sigmoid")(base_model)
    # Fully connected -> 512
    # batch norm
    # dropout
    # relu
    attributes_weight = Dense(n_attributes, activation="sigmoid")(attribute_layer)
    reweighted_attribute = Multiply()([attribute_layer, attributes_weight])

    id_layer = Dense(512, activation="relu")(base_model)
    id_layer = BatchNormalization()(id_layer)
    id_layer = Dropout(rate=0.5)(id_layer)
    id_layer = Concatenate(axis=-1, name="image_features_concat")([reweighted_attribute, id_layer])

    training_id_layer = Dense(n_person, activation="softmax")(id_layer)

    attribute_model = Model(input_tensor, attribute_layer)
    person_id_model = Model(input_tensor, id_layer)
    training_model = Model(input_tensor, [attribute_layer, training_id_layer])
    # binary, categorical

    return (attribute_model, training_model, person_id_model)

# loss binary crossentropy?
# age not binary
attributes: np.ndarray = np.array([
    'age', 'backpack', 'bag', 'clothes', 'down', 'downblack',
    'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink',
    'downpurple', 'downwhite', 'downyellow', 'gender', 'hair',
    'handbag', 'hat', 'up', 'upblack', 'upblue', 'upgray', 'upgreen',
    'uppurple', 'upred', 'upwhite', 'upyellow'], dtype=object)

down_colors = slice(5, 14)
up_colors = slice(19, 27)
binary_1 = slice(0, 5)
binary_2 = slice(14, 19)

def market_attribute_accuracy(y_true, y_pred):
    # binary categories
    acc = binary_accuracy(y_true[:, binary_1], y_pred[:, binary_1]) * 0.5
    acc += binary_accuracy(y_true[:, binary_2], y_pred[:, binary_2]) * 0.5
    # top colors
    acc_top_color = categorical_accuracy(y_true[:, up_colors], y_pred[:, up_colors])
    # down colors
    acc_down_color = categorical_accuracy(y_true[:, down_colors], y_pred[:, down_colors])
    
    # print(acc, acc_down_color, acc_top_color)
    return acc * 9/11 + acc_down_color * 1/11 + acc_top_color * 1/11

# def i(a):return a
if __name__ == "__main__":
    train_ms = MarketSequence("Market-1501/market.h5", 128, preprocess=(preprocess_images, preprocess_labels))
    # TODO: callbacks
    model, _, _ = get_models(n_attributes=26)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", market_attribute_accuracy])
    # model.summary()
    # test_ms = MarketSequence("market.h5", 1024, train=False, preprocess=(resize_preprocess, preprocess_labels))
    hist = model.fit(preprocess_images(np.array(train_ms.x)), preprocess_labels(np.array(train_ms.y)), batch_size=32, validation_split=0.2)
    # hist = model.fit_generator(train_ms, epochs=1)
    # model.fit(*train_ms[0], batch_size=16)
    # print(model.evaluate(*test_ms[0]))
    print(hist.history)
    # model.save("model_attr.h5")
