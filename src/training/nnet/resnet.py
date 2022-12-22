# ローカルだとtensorflow 2.5系以降しかインストールできないためimportに失敗するが、Docker上では2.3系のため問題なく動作する
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras import layers, Sequential, Model

from src.training.nnet.base_nnet import BaseNNet


class ResNet(BaseNNet):
    def build_model(self):
        """
        モデルをビルド
        """

        input_tensor = layers.Input(shape=(64, 64, 1))
        resnet = ResNet50V2(weights=None, input_tensor=input_tensor, include_top=False)

        top_model = Sequential()
        top_model.add(layers.Flatten(input_shape=resnet.output_shape[1:]))
        top_model.add(layers.Dense(6, activation=self.output_activation))
        self.model = Model(inputs=resnet.inputs, outputs=top_model(resnet.outputs))
