from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Model, metrics
from tensorflow.python.keras.callbacks import ModelCheckpoint
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
            metrics=[self.x_mae, self.y_mae, self.σ_x_mae, self.σ_y_mae],
        )

    def load_weights(self, filename):
        """
        訓練済みモデルをロード
        """

        self.model.load_weights(filename)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        """
        訓練
        """

        # チェックポイントを保存するコールバックを定義
        checkpoint_filename: str = "./ckpt/{epoch}.h5"
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filename,
            verbose=1,
            save_weights_only=True
        )
        self.model.save_weights(checkpoint_filename.format(epoch=0))

        # early stopping
        # early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

        self.compile_model()
        return self.model.fit(
            x=x_train,
            y=y_train,
            epochs=5,
            batch_size=64,
            validation_data=(x_test, y_test),
            # callbacks=[checkpoint_callback, early_stopping_callback],
            callbacks=[checkpoint_callback],
        )

    def loss(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        損失関数
        """

        # y = [xの移動量, yの移動量] ^ T
        y = tf.expand_dims(y_true[:, 0:2], axis=-1)

        # ŷ = [推定したxの移動量, 推定したyの移動量] ^ T
        ŷ = tf.expand_dims(y_pred[:, 0:2], axis=-1)

        return tf.reduce_mean(
            tf.losses.mean_squared_error(y, ŷ)
        )

        Σ = self.get_Σ(y_pred)

        return tf.reduce_mean(
            tf.math.log((2 * np.pi) ** 2 * tf.linalg.det(Σ))
            + tf.matmul(tf.matmul(tf.linalg.matrix_transpose(y - ŷ), tf.linalg.inv(Σ)), (y - ŷ))
        )

    @staticmethod
    def output_activation(y_pred: np.ndarray) -> Tensor:
        """
        出力層の活性化関数
        """

        ŷ1 = y_pred[:, 0]
        ŷ2 = y_pred[:, 1]
        ρ1 = y_pred[:, 2]
        ρ2 = y_pred[:, 3]
        θ = y_pred[:, 4]

        return tf.stack([ŷ1, ŷ2, ρ1, ρ2, θ], axis=1)

    @staticmethod
    def x_mae(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        xの移動量の評価関数
        """

        y1 = y_true[:, 0]
        ŷ1 = y_pred[:, 0]

        return metrics.mean_absolute_error(y1, ŷ1)

    @staticmethod
    def y_mae(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        yの移動量の評価関数
        """

        y_2 = y_true[:, 1]
        ŷ_2 = y_pred[:, 1]

        return metrics.mean_absolute_error(y_2, ŷ_2)

    def σ_x_mae(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        xの移動量の標準偏差の評価関数
        """

        y1 = y_true[:, 0]
        ŷ1 = y_pred[:, 0]

        Σ = self.get_Σ(y_pred)
        σ_x = K.sqrt(Σ[:, 0, 0])

        return metrics.mean_absolute_error(K.abs(y1 - ŷ1), σ_x)

    def σ_y_mae(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        xの移動量の標準偏差の評価関数
        """

        y2 = y_true[:, 0]
        ŷ2 = y_pred[:, 0]

        Σ = self.get_Σ(y_pred)
        σ_y = K.sqrt(Σ[:, 1, 1])

        return metrics.mean_absolute_error(K.abs(y2 - ŷ2), σ_y)

    @staticmethod
    def get_Σ(y_pred: Tensor):
        """
        共分散行列を取得
        """

        # Σ = U * Λ * U^T のように分解する（Λは対角行列、Uは回転行列）
        # Λ = [[λ1, 0], [0, λ2]] = [[e^ρ1, 0], [0, e^ρ2]]
        # λ>=0を満たすために、λ=e^ρで変数変換する（NNはρを出力）
        Λ = tf.linalg.diag(tf.exp(y_pred[:, 2:4]))

        # U = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
        θ = y_pred[:, 4]
        sin_θ = tf.expand_dims(K.sin(θ), axis=-1)
        cos_θ = tf.expand_dims(K.cos(θ), axis=-1)
        U = tf.concat([
            tf.expand_dims(tf.concat([cos_θ, sin_θ], axis=-1), axis=-1),
            tf.expand_dims(tf.concat([-sin_θ, cos_θ], axis=-1), axis=-1),
        ], axis=-1)

        # Σ = U * Λ * U^T
        # FIXME: InvalidArgumentError: Input is not invertible.
        return tf.linalg.matmul(tf.linalg.matmul(U, Λ), tf.linalg.matrix_transpose(U))

    def predict(self, x: np.ndarray):
        """
        相対位置ベクトルを推定
        """

        return self.model.predict(x)
