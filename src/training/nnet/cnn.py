from tensorflow.python.keras import Sequential, layers

from src.training.nnet.base_nnet import BaseNNet


class CNN(BaseNNet):
    def build_model(self):
        """
        モデルをビルド
        """

        self.model = Sequential()

        self.model.add(layers.Conv2D(32, 3, input_shape=(64, 64, 1)))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.Conv2D(32, 3))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))

        self.model.add(layers.Conv2D(64, 3))
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPool2D(pool_size=(2, 2)))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1024))
        self.model.add(layers.Activation('relu'))

        self.model.add(layers.Dense(5, activation=self.output_activation))
