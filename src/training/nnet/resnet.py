# ローカルだとtensorflow 2.5系以降しかインストールできないためimportに失敗するが、Docker上では2.3系のため問題なく動作する
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.python.keras import layers, Sequential, Model

from src.training.nnet.base_nnet import BaseNNet


class ResNet(BaseNNet):
    def build_model(self):
        """
        モデルをビルド
        """

        input_tensor = layers.Input(shape=(128, 128, 1))
        input_concatenate = layers.Concatenate()([input_tensor, input_tensor, input_tensor])
        resnet = ResNet50(input_tensor=input_concatenate, include_top=True)

        top_model = Sequential()
        top_model.add(layers.Flatten(input_shape=resnet.output_shape[1:]))
        top_model.add(layers.Dense(5, activation=self.output_activation))
        self.model = Model(inputs=resnet.inputs, outputs=top_model(resnet.outputs))
