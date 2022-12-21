# ローカルだとtensorflow 2.5系以降しかインストールできないためimportに失敗するが、Docker上では2.3系のため問題なく動作する
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras import layers, Sequential

from src.training.nnet.base_nnet import BaseNNet


class ResNet(BaseNNet):
    def build_model(self):
        """
        モデルをビルド
        """

        self.model = Sequential([
            ResNet50V2(weights=None, input_shape=(500, 500, 1), include_top=False),
            layers.Dense(5, activation=self.output_activation)
        ])
