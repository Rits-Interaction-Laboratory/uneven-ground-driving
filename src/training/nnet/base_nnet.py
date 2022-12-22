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
        # TODO: 削除する
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
            batch_size=128,
            validation_split=0.1,
        )

    @staticmethod
    def loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        損失関数
        """

        # 分母が極端に小さくなることを防ぐためのオフセット
        epsilon = K.constant(K.epsilon())

        # y: 実際の相対移動ベクトル（列ベクトル）
        y = tf.linalg.matrix_transpose(y_true)

        # θ: 推定した相対移動ベクトル（列ベクトル）
        θ = tf.linalg.matrix_transpose(y_true[:, 0:2])

        # Σ = U * Λ * U^T のように分解する（ただし、Λは対角行列、Uは回転行列）
        # Λ = [[λ1, 0], [0, λ2]]
        # Λ^1 = [[1/λ1, 0], [0, 1/λ2]]
        Λ = tf.linalg.diag(y_pred[:, 2:4])
        Λ_inv = tf.linalg.diag(1.0 / (y_pred[:, 2:4] + epsilon))

        # U = [[u1, u2], [u2, -u1]]
        # 下記より、u1からUを求められる（NNはu1のみ出力する）
        #   1. 1列目と2列目はそれぞれ単位ベクトル（=u2が一意に定まる）
        #   2. 1列目と2列目は直交する
        u1 = K.reshape(y_pred[:, 4], (y_pred.shape[0], 1))
        u2 = K.reshape(y_pred[:, 5], (y_pred.shape[0], 1))
        U = K.concatenate([K.reshape(K.concatenate([u1, u2]), (y_pred.shape[0], 2, 1)),
                           K.reshape(K.concatenate([-u2, u1]), (y_pred.shape[0], 2, 1))])

        # Σ = U * Λ * U^T
        # Σ^-1 = U * Λ^1 * U^T
        Σ = tf.matmul(tf.matmul(U, Λ), tf.linalg.matrix_transpose(U)) + epsilon
        Σ_inv = tf.matmul(tf.matmul(U, Λ_inv), tf.linalg.matrix_transpose(U)) + epsilon
        det_Σ = Σ[:, 0, 0] * Σ[:, 1, 1] - Σ[:, 0, 1] * Σ[:, 1, 0]

        return K.mean(
            K.log((2 * np.pi) ** 2 * (det_Σ + epsilon)) + \
            tf.matmul(tf.matmul(tf.linalg.matrix_transpose(y - θ), Σ_inv), (y - θ))
        )

    @staticmethod
    def output_activation(y_pred: np.ndarray) -> Tensor:
        """
        出力層の活性化関数
        """

        θ_x = K.sigmoid(y_pred[:, 0])
        θ_y = K.sigmoid(y_pred[:, 1])
        λ1 = K.relu(y_pred[:, 2])
        λ2 = K.relu(y_pred[:, 3])
        u1 = y_pred[:, 4]
        u2 = y_pred[:, 5]
        u1_u2_vector_length = K.sqrt(u1 ** 2 + u2 ** 2)
        u1 /= u1_u2_vector_length
        u2 /= u1_u2_vector_length

        return K.stack([θ_x, θ_y, λ1, λ2, u1, u2], 1)
