from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Model, metrics
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
        # tf.config.run_functions_eagerly(True)
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
            metrics=[self.x_movement_amount_metric, self.y_movement_amount_metric],
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
        epsilon = tf.constant(K.epsilon())

        # y = [xの移動量, yの移動量] ^ T
        y = tf.expand_dims(y_true[:, 0:2], axis=-1)

        # ŷ = [推定したxの移動量, 推定したyの移動量] ^ T
        ŷ = tf.expand_dims(y_pred[:, 0:2], axis=-1)

        # Σ = U * Λ * U^T のように分解する（Λは対角行列、Uは回転行列）
        # Λ = [[λ1, 0], [0, λ2]]
        # Λ^1 = [[1/λ1, 0], [0, 1/λ2]] = [[ξ1, 0], [0, ξ2]]
        Λ_inv = tf.linalg.diag(y_pred[:, 2:4] + epsilon)

        # U = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        θ = y_pred[:, 4]
        sin_θ = tf.expand_dims(K.sin(θ), axis=-1)
        cos_θ = tf.expand_dims(K.cos(θ), axis=-1)
        U = tf.concat([
            tf.expand_dims(tf.concat([cos_θ, sin_θ], axis=-1), axis=-1),
            tf.expand_dims(tf.concat([-sin_θ, cos_θ], axis=-1), axis=-1),
        ], axis=-1)

        # Σ = U * Λ * U^T
        # Σ^-1 = U * Λ^1 * U^T
        Σ_inv = tf.matmul(tf.matmul(U, Λ_inv), tf.linalg.matrix_transpose(U))
        det_Σ = 1.0 / (tf.linalg.det(Σ_inv) + epsilon)

        return tf.reduce_mean(
            tf.math.log((2 * np.pi) ** 2 * det_Σ) + \
            tf.matmul(tf.matmul(tf.linalg.matrix_transpose(y - ŷ), Σ_inv), (y - ŷ))
        )

    @staticmethod
    def output_activation(y_pred: np.ndarray) -> Tensor:
        """
        出力層の活性化関数
        """

        ŷ1 = y_pred[:, 0]
        ŷ2 = y_pred[:, 1]
        ξ1 = tf.nn.relu(y_pred[:, 2])
        ξ2 = tf.nn.relu(y_pred[:, 3])
        θ = y_pred[:, 4]

        return tf.stack([ŷ1, ŷ2, ξ1, ξ2, θ], axis=1)

    @staticmethod
    def x_movement_amount_metric(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        xの移動量の評価関数
        """

        y1 = y_true[:, 0]
        ŷ1 = y_pred[:, 0]

        return metrics.mean_absolute_error(y1, ŷ1)

    @staticmethod
    def y_movement_amount_metric(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        yの移動量の評価関数
        """

        y_2 = y_true[:, 1]
        ŷ_2 = y_pred[:, 1]

        return metrics.mean_absolute_error(y_2, ŷ_2)
