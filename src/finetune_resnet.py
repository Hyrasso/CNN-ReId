# import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import Input, Model
from keras.metrics import binary_accuracy, categorical_accuracy
from keras.layers import Dense, Lambda, MaxPooling2D, Flatten, Concatenate, BatchNormalization, Dropout, Multiply
import keras.backend as K
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

def get_models(n_person: int=1501, n_attributes: int=27) -> Model:

    input_tensor = Input(shape=(128, 64, 3))  # channels_last

    # create the base pre-trained model
    base_model = ResNet50(input_shape=(128, 64, 3), pooling="avg", weights='imagenet',include_top=False)(input_tensor)

    attribute_layer = Dense(n_attributes, activation="sigmoid")(base_model)
    # Fully connected -> 512
    # batch norm
    # dropout
    # relu
    attributes_weight = Dense(n_attributes, activation="sigmoid", use_bias=False)(attribute_layer)
    reweighted_attribute = Multiply()([attribute_layer, attributes_weight])

    id_layer = Dense(512, activation="relu")(base_model)
    id_layer = BatchNormalization()(id_layer)
    id_layer = Dropout(0.5)(id_layer)
    id_layer = Concatenate(axis=-1, name="image_features_concat")([reweighted_attribute, id_layer])

    training_id_layer = Dense(n_person, activation="softmax")

    attribute_model = Model(input_tensor, attribute_layer)
    person_id_model = Model(input_tensor, id_layer)
    training_model = Model(input_tensor, [attribute_layer, training_id_layer])
    # binary, categorical

    return (training_model, attribute_model, person_id_model)

# loss binary crossentropy?
# age not binary

def market_attribute_accuracy(y_true, y_pred):
    binary_categories = slice(0, 9)
    top_colors = slice(9, 17)
    down_colors = slice(17, 26)
    # cat
    acc = binary_accuracy(y_true[binary_categories], y_pred[binary_categories])
    # top colors
    acc_top_color = categorical_accuracy(y_true[top_colors], y_pred[top_colors])
    # down colors
    acc_down_color = categorical_accuracy(y_true[down_colors], y_pred[down_colors])
    
    print(acc, acc_down_color, acc_top_color)
    return acc * 9/11 + acc_down_color * 1/11 + acc_top_color * 1/11

# def i(a):return a
if __name__ == "__main__":
    train_ms = MarketSequence("Market-1501/market.h5", 128, preprocess=(resize_preprocess, preprocess_labels))

    _, model, _ = get_models(n_attributes=26)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # model.summary()
    # test_ms = MarketSequence("market.h5", 1024, train=False, preprocess=(resize_preprocess, preprocess_labels))
    hist = model.fit(resize_preprocess(np.array(train_ms.x)), preprocess_labels(np.array(train_ms.y)), batch_size=32, validation_split=0.2)
    # hist = model.fit_generator(train_ms, epochs=1)
    # model.fit(*train_ms[0], batch_size=16)
    # print(model.evaluate(*test_ms[0]))
    print(hist.history)
    # model.save("model_attr.h5")
