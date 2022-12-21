from abc import ABCMeta, abstractmethod

import numpy as np
from tensorflow.python.keras import Model, losses
from tensorflow.python.types.core import Tensor


class BaseNNet(metaclass=ABCMeta):
    """
    ニューラルネットワークの基底クラス
    """

    model: Model
    """
    モデル
    """

    def __init__(self):
        self.build_model()
        self.model.summary()

    @abstractmethod
    def build_model(self):
        """
        モデルをビルド
        """

        raise NotImplementedError()

    def compile_model(self):
        """
        モデルをコンパイル
        """

        self.model.compile(
            optimizer="adam",
            loss=self.loss,
            metrics=[],
        )

    def load_weights(self, filename):
        """
        訓練済みモデルをロード
        """

        self.model.load_weights(filename)

    def train(self, x: np.ndarray, y: np.ndarray):
        """
        訓練
        """

        self.compile_model()
        self.model.fit(
            x=x,
            y=y,
            epochs=50,
            batch_size=256,
            validation_split=0.1,
        )

    @staticmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> Tensor:
        """
        損失関数
        """

        # TODO: 損失関数を定義する
        return losses.mean_squared_error(y_true, y_pred)
