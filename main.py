import logging
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

from src.training.driving_record import DrivingRecord, DrivingRecordRepository
from src.training.nnet.base_nnet import BaseNNet
from src.training.nnet.resnet import ResNet

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s --- %(message)s',
)

nnet: BaseNNet = ResNet()
driving_record_repository: DrivingRecordRepository = DrivingRecordRepository()


def output_stats(data: list[DrivingRecord]):
    """
    統計を出力
    """

    movement_amounts: list[tuple[float, float]] = [driving_record.get_movement_amount() for driving_record in data]
    logging.info(f'件数: {len(movement_amounts)}')
    logging.info(f' x の平均: {np.mean([movement_amount[0] for movement_amount in movement_amounts])}')
    logging.info(f' y の平均: {np.mean([movement_amount[1] for movement_amount in movement_amounts])}')
    logging.info(f'|x|の平均: {np.mean([abs(movement_amount[0]) for movement_amount in movement_amounts])}')
    logging.info(f'|y|の平均: {np.mean([abs(movement_amount[1]) for movement_amount in movement_amounts])}')

    covariance_matrix: np.ndarray = np.cov(movement_amounts, rowvar=False)
    for line in str(covariance_matrix).split('\n'):
        logging.info(line)

    # X、Yの移動量のヒストグラムを生成
    x_movement_amounts = [movement_amount[0] for movement_amount in movement_amounts]
    y_movement_amounts = [movement_amount[1] for movement_amount in movement_amounts]

    os.makedirs("./analysis", exist_ok=True)
    plt.figure()
    plt.hist(x_movement_amounts, bins=100)
    plt.savefig("./analysis/histogram_x.png")

    plt.figure()
    plt.hist(y_movement_amounts, bins=100)
    plt.savefig("./analysis/histogram_y.png")

    plt.figure()
    plt.hist2d(x_movement_amounts, y_movement_amounts, bins=100, cmin=1)
    plt.colorbar()
    plt.savefig("./analysis/histogram_xy.png")


def output_predict_results(data: list[DrivingRecord]):
    """
    推定結果を出力
    """

    predict_results = nnet.predict(
        np.array([driving_record.image for driving_record in driving_records], dtype=np.float32))

    pred_x_movement_amounts = predict_results[:, 0]
    pred_y_movement_amounts = predict_results[:, 1]
    true_x_movement_amounts = [driving_record.get_movement_amount()[0] for driving_record in data]
    true_y_movement_amounts = [driving_record.get_movement_amount()[1] for driving_record in data]

    plt.figure()
    plt.hist2d(pred_x_movement_amounts, true_x_movement_amounts, bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel("ŷ_0")
    plt.ylabel("y_0")
    plt.savefig("./analysis/heatmap_x.png")

    plt.figure()
    plt.hist2d(pred_y_movement_amounts, true_y_movement_amounts, bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel("ŷ_1")
    plt.ylabel("y_1")
    plt.savefig("./analysis/heatmap_y.png")


logging.info('訓練データセットをロード開始')
driving_records: list[DrivingRecord] = driving_record_repository.select_all()
output_stats(driving_records)
x: np.ndarray = np.array([driving_record.image for driving_record in driving_records], dtype=np.float32)
y: np.ndarray = np.array([driving_record.get_movement_amount() for driving_record in driving_records],
                         dtype=np.float32)
# y = (y - y.min()) / (y.max() - y.min())

logging.info('訓練開始')
history = nnet.train(x, y)

# 訓練履歴をプロット
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.savefig("./analysis/loss_history.png")

output_predict_results(driving_records)
