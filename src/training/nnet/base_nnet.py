from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Model
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
        tf.config.run_functions_eagerly(True)
        self.build_model()
        # self.model.summary()

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
    def loss(y_true_tensor: Tensor, y_pred_tensor: Tensor) -> Tensor:
        """
        損失関数
        """

        # 分母が極端に小さくなることを防ぐためのオフセット
        epsilon = K.constant(K.epsilon())

        losses: list[Tensor] = []
        for y_true, y_pred in zip(y_true_tensor, y_pred_tensor):
            # y: 実際の相対移動ベクトル（列ベクトル）
            y = K.reshape(y_true, (2, 1))

            # θ: 推定した相対移動ベクトル（列ベクトル）
            θ = K.reshape(y_pred[0:2], (2, 1))

            # Σ = U * Λ * U^T のように分解する（ただし、Λは対角行列、Uは回転行列）
            # Λ = [[λ1, 0], [0, λ2]]
            # Λ^1 = [[1/λ1, 0], [0, 1/λ2]]
            λ1 = y_pred[2]
            λ2 = y_pred[3]
            Λ = tf.linalg.diag([λ1, λ2])
            Λ_inv = tf.linalg.diag([1.0 / (λ1 + epsilon), 1.0 / (λ2 + epsilon)])

            # U = [[u1, u2], [u2, -u1]]
            # 下記より、u1からUを求められる（NNはu1のみ出力する）
            #   1. 1列目と2列目はそれぞれ単位ベクトル（=u2が一意に定まる）
            #   2. 1列目と2列目は直交する
            u1 = y_pred[4] + epsilon
            u2 = K.sqrt(K.constant(1) - u1 ** 2)
            U = K.variable([[u1, u2], [u2, -u1]])

            # Σ = U * Λ * U^T
            # Σ^-1 = U * Λ^1 * U^T
            Σ = K.dot(K.dot(U, Λ), K.transpose(U))
            Σ_inv = K.dot(K.dot(U, Λ_inv), K.transpose(U))

            loss = K.log((2 * np.pi) ** 2 * tf.linalg.det(Σ)) + \
                   K.dot(K.dot(K.transpose(y - θ), Σ_inv), (y - θ))
            losses.append(loss)

        return K.mean(K.constant([loss.numpy() for loss in losses]))

    @staticmethod
    def output_activation(y_pred: np.ndarray) -> Tensor:
        """
        出力層の活性化関数
        """

        θ_x = y_pred[:, 0]
        θ_y = y_pred[:, 1]
        λ1 = K.relu(y_pred[:, 2])
        λ2 = K.relu(y_pred[:, 3])
        u1 = K.sigmoid(y_pred[:, 4])

        return K.stack([θ_x, θ_y, λ1, λ2, u1], 1)
