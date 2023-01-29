import os

import matplotlib.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from src.training.driving_record import DrivingRecord, DrivingRecordRepository
from src.training.nnet.base_nnet import BaseNNet
from src.training.nnet.cnn import CNN

driving_record_repository: DrivingRecordRepository = DrivingRecordRepository()
driving_records: list[DrivingRecord] = driving_record_repository.select_all()

split_index: int = int(len(driving_records) // 10)
train_driving_records = driving_records[split_index:]
test_driving_records = driving_records[0:split_index]

# 過去の推定結果を読み込む
train_predict_results: np.ndarray = np.load("./analysis/train_snapshot.npy")
test_predict_results: np.ndarray = np.load("./analysis/test_snapshot.npy")

# 過去の推定結果を流用するので、読み込みに時間のかかるResNetは使わない
nnet: BaseNNet = CNN()
train_Σ_list = nnet.get_Σ(train_predict_results)
test_Σ_list = nnet.get_Σ(test_predict_results)


def confidence_ellipse(x_mean, y_mean, covariance_matrix, ax, n_std=2):
    pearson = covariance_matrix[0, 1] / np.sqrt(covariance_matrix[0, 0] * covariance_matrix[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, color='blue', fill=False)

    scale_x = np.sqrt(covariance_matrix[0, 0]) * n_std
    scale_y = np.sqrt(covariance_matrix[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x_mean, y_mean)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

    return pearson


# 各サンプルの(x - x̂)を中心とする誤差楕円を描画
for i in range(len(test_driving_records)):
    driving_record = test_driving_records[i]
    predict_result = test_predict_results[i]
    Σ = test_Σ_list[i]

    x: np.ndarray = np.array(driving_record.get_movement_amount(), dtype=np.float32)
    x̂: np.ndarray = np.array(predict_result[:2], dtype=np.float32)

    fig = plt.figure()
    ax = fig.add_subplot()

    confidence_ellipse(x̂[0], x̂[1], Σ, ax, 1)
    confidence_ellipse(x̂[0], x̂[1], Σ, ax, 2)

    ax.scatter([x̂[0]], [x̂[1]], label=r"$\hat{\mathbf{x}}$")
    ax.scatter([x[0]], [x[1]], label=r"$\mathbf{x}$")
    ax.legend()
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim(0.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.savefig(f"./analysis/error_ellipse/test/{os.path.basename(driving_record.image_filename)}")
    plt.close()
