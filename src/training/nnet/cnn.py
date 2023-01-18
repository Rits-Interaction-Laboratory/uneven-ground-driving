from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras import Sequential, layers

from src.training.nnet.base_nnet import BaseNNet


class CNN(BaseNNet):
    def build_model(self):
        """
        モデルをビルド
        """

        self.model = Sequential()

        # input layer
        self.model.add(layers.Input(shape=(128, 128, 3)))

        # convolution 1st layer
        self.model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(layers.MaxPool2D())

        # convolution 2nd layer
        self.model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(layers.MaxPool2D())

        # fully connected 1st layer
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation="relu"))
        self.model.add(BatchNormalization())

        # fully connected final layer
        self.model.add(layers.Dense(5, activation=self.output_activation))
