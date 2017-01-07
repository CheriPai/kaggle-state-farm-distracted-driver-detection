import numpy as np
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential


class VGG16():

    weights_path = "data/vgg16_bn.h5"
    model = None

    def __init__(self):
        self.build()

    def vgg_preprocess(self, x):
        vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1))
        x = x - vgg_mean
        return x[:, ::-1]

    def conv_block(self, layers, filters):
        for i in range(layers):
            self.model.add(ZeroPadding2D((1, 1)))
            self.model.add(Convolution2D(filters, 3, 3, activation="relu"))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def fc_block(self, dropout):
        self.model.add(Dense(4096, activation="relu"))
        self.model.add(Dropout(dropout))
        self.model.add(BatchNormalization())

    def build(self, dropout=0.3):
        self.model = Sequential()
        self.model.add(Lambda(self.vgg_preprocess, input_shape=(3, 244, 244)))

        self.conv_block(2, 64)
        self.conv_block(2, 128)
        self.conv_block(3, 256)
        self.conv_block(3, 512)
        self.conv_block(3, 512)
        self.model.add(Flatten())

        self.fc_block(dropout)
        self.fc_block(dropout)
        self.model.add(Dense(1000, activation="softmax"))
        self.model.load_weights(self.weights_path)


if __name__ == "__main__":
    vgg16 = VGG16()
    print(vgg16.model.summary())
