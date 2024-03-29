import logging
import os
import sys

import numpy as np
import tensorflow.python.keras.backend as K
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
    plt.xlabel(r"$x_1$")
    plt.ylabel("samples")
    plt.savefig("./analysis/histogram_x.png")

    plt.figure()
    plt.hist(y_movement_amounts, bins=100)
    plt.xlabel(r"$x_2$")
    plt.ylabel("samples")
    plt.savefig("./analysis/histogram_y.png")

    plt.figure()
    plt.hist2d(x_movement_amounts, y_movement_amounts, bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.savefig("./analysis/histogram_xy.png")


def output_train_history(history):
    """
    訓練履歴を出力
    """

    plt.figure()
    plt.plot(np.log(history.history['loss']))
    plt.plot(np.log(history.history['val_loss']))
    plt.ylabel('log(loss)')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig("./analysis/history_loss.png")

    plt.figure()
    plt.plot(history.history['x_mae'])
    plt.plot(history.history['val_x_mae'])
    plt.ylabel('|x_1 - x̂_1|')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig("./analysis/history_x_mae.png")

    plt.figure()
    plt.plot(history.history['y_mae'])
    plt.plot(history.history['val_y_mae'])
    plt.ylabel('|x_2 - x̂_2|')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig("./analysis/history_y_mae.png")

    # plt.figure()
    # plt.plot(history.history['σ_x_mae'])
    # plt.plot(history.history['val_σ_x_mae'])
    # plt.ylabel('||x_1 - x̂_1| - σ_y1|')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'])
    # plt.savefig("./analysis/history_σ_x_mae.png")

    # plt.figure()
    # plt.plot(history.history['σ_y_mae'])
    # plt.plot(history.history['val_σ_y_mae'])
    # plt.ylabel('||x_2 - x̂_2| - σ_y2|')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'])
    # plt.savefig("./analysis/history_σ_y_mae.png")


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def output_predict_results(predict_results: np.ndarray, true_movement_amounts: np.ndarray, label: str,
                           heatmap_y1_range: tuple, heatmap_y2_range: tuple, heatmap_σ1_range: tuple,
                           heatmap_σ2_range: tuple, y1_error_range: tuple, y2_error_range: tuple):
    """
    推定結果を出力
    """

    np.save(f"analysis/{label}_snapshot.npy", predict_results)

    ŷ1_movement_amounts = predict_results[:, 0]
    ŷ2_movement_amounts = predict_results[:, 1]
    y1_movement_amounts = true_movement_amounts[:, 0]
    y2_movement_amounts = true_movement_amounts[:, 1]

    # TODO: ヒートマップ上にy=xの直線を表示
    plt.figure()
    plt.hist2d(ŷ1_movement_amounts, y1_movement_amounts, bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel(r"$\hat{x_1}$")
    plt.ylabel(r"$x_1$")
    plt.xlim(*heatmap_y1_range)
    plt.ylim(0.0, 0.9)
    plt.savefig(f"./analysis/{label}_heatmap_x.png")

    plt.figure()
    plt.hist2d(ŷ2_movement_amounts, y2_movement_amounts, bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel(r"$\hat{x_2}$")
    plt.ylabel(r"$x_2$")
    plt.xlim(*heatmap_y2_range)
    plt.ylim(-0.4, 0.4)
    plt.savefig(f"./analysis/{label}_heatmap_y.png")

    """
    # 真値と推定値の線分を描画
    lines = [[[y1_movement_amounts[i], y2_movement_amounts[i]], [ŷ1_movement_amounts[i], ŷ2_movement_amounts[i]]]
             for i in range(predict_results.shape[0])]
    for size in [100, 200, 300]:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.add_collection(LineCollection(lines[:size]))
        ax.autoscale()
        plt.savefig(f"./analysis/{label}_y_ŷ_lines_{size}.png")

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.add_collection(LineCollection(lines[:size]))
        ax.set_xlim(0.65, 0.83)
        ax.set_ylim(-0.18, 0.18)
        plt.savefig(f"./analysis/{label}_y_ŷ_lines_expansion_{size}.png")

    Σ_list = nnet.get_Σ(predict_results)

    # 推定値を中心とする共分散行列に対応する楕円を描画
    nstd = 2
    for size in range(10, 301, 50):
        fig = plt.figure()
        ax = fig.add_subplot()

        for i in range(len(predict_results[:size])):
            vals, vecs = eigsorted(Σ_list[i])
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * nstd * np.sqrt(vals)
            ell = Ellipse(xy=(ŷ1_movement_amounts[i], ŷ2_movement_amounts[i]),
                          width=w, height=h,
                          angle=theta, color='black')
            ell.set_facecolor('none')
            ax.add_artist(ell)

        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        plt.savefig(f"./analysis/{label}_Σ_{size}.png")
    """

    Σ_list = nnet.get_Σ(predict_results)
    σ1_list = K.sqrt(Σ_list[:, 0, 0])
    σ2_list = K.sqrt(Σ_list[:, 1, 1])

    y1_errors = []
    y2_errors = []
    for i in range(len(predict_results)):
        y1_errors.append(abs(y1_movement_amounts[i] - ŷ1_movement_amounts[i]))
        y2_errors.append(abs(y2_movement_amounts[i] - ŷ2_movement_amounts[i]))

    plt.figure()
    plt.hist2d(σ1_list, y1_errors, bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel(r"$\hat{\sigma_1}$")
    plt.ylabel(r"$|\hat{x_1} - x_1|$")
    plt.savefig(f"./analysis/{label}_heatmap_σ_x_error.png")

    plt.figure()
    plt.hist2d(σ2_list, y2_errors, bins=100, cmin=1)
    plt.colorbar()
    plt.xlabel(r"$\hat{\sigma_2}$")
    plt.ylabel(r"$|\hat{x_2} - x_2|$")
    plt.savefig(f"./analysis/{label}_heatmap_σ_y_error.png")

    # σと残差標準偏差をプロットする（ただし、残差標準偏差を求める際は平均で引かない）
    σ1_standard_deviation_graph_y_list: list[float] = []
    σ1_standard_deviation_graph_x_list: list[float] = []
    σ2_standard_deviation_graph_y_list: list[float] = []
    σ2_standard_deviation_graph_x_list: list[float] = []

    σ_bins: int = 200
    σ1_range_width = (heatmap_σ1_range[1] - heatmap_σ1_range[0]) / σ_bins
    σ2_range_width = (heatmap_σ2_range[1] - heatmap_σ2_range[0]) / σ_bins
    for i in range(-1, σ_bins + 1):
        y1_start = heatmap_σ1_range[0] + σ1_range_width * i
        y1_end = y1_start + σ1_range_width
        y2_start = heatmap_σ2_range[0] + σ2_range_width * i
        y2_end = y2_start + σ2_range_width

        σ1_sum: float = 0.0
        σ1_cnt: int = 0
        σ2_sum: float = 0.0
        σ2_cnt: int = 0
        for j in range(len(predict_results)):
            if y1_start <= σ1_list[j] < y1_end:
                σ1_sum += y1_errors[j] ** 2
                σ1_cnt += 1
            if y2_start <= σ2_list[j] < y2_end:
                σ2_sum += y2_errors[j] ** 2
                σ2_cnt += 1

        for _ in range(σ1_cnt):
            σ1_standard_deviation_graph_x_list.append(y1_start)
            σ1_standard_deviation_graph_y_list.append(np.sqrt(σ1_sum / σ1_cnt))
        for _ in range(σ2_cnt):
            σ2_standard_deviation_graph_x_list.append(y2_start)
            σ2_standard_deviation_graph_y_list.append(np.sqrt(σ2_sum / σ2_cnt))

    plt.figure()
    plt.hist2d(σ1_standard_deviation_graph_x_list, σ1_standard_deviation_graph_y_list, bins=σ_bins, cmin=1)
    plt.colorbar()
    plt.xlabel(r"$\hat{\sigma_1}$")
    plt.ylabel("RMSE")
    plt.xlim(*heatmap_σ1_range)
    plt.ylim(*y1_error_range)
    plt.savefig(f"./analysis/{label}_heatmap_σ_x_standard_deviation.png")

    plt.figure()
    plt.hist2d(σ2_standard_deviation_graph_x_list, σ2_standard_deviation_graph_y_list, bins=σ_bins, cmin=1)
    plt.colorbar()
    plt.xlabel(r"$\hat{\sigma_2}$")
    plt.ylabel("RMSE")
    plt.xlim(*heatmap_σ2_range)
    plt.ylim(*y2_error_range)
    plt.savefig(f"./analysis/{label}_heatmap_σ_y_standard_deviation.png")


logging.info('訓練データセットをロード開始')
driving_records: list[DrivingRecord] = driving_record_repository.select_all()
output_stats(driving_records)
x: np.ndarray = np.array([driving_record.image for driving_record in driving_records], dtype=np.float32)
y: np.ndarray = np.array([driving_record.get_movement_amount() for driving_record in driving_records],
                         dtype=np.float32)

split_index: int = int(len(driving_records) // 10)
x_train: np.ndarray = x[split_index:]
x_test: np.ndarray = x[0:split_index]
y_train: np.ndarray = y[split_index:]
y_test: np.ndarray = y[0:split_index]

logging.info('訓練開始')
history = nnet.train(x_train, y_train)

# 訓練履歴をプロット
output_train_history(history)

# 最良エポックでの推定結果をプロット
best_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
logging.info(f"最良エポック：{best_epoch}")
nnet.load_weights(f"ckpt/{best_epoch}.h5")

train_predict_results = nnet.predict(
    np.array([driving_record.image for driving_record in driving_records[split_index:]], dtype=np.float32))
test_predict_results = nnet.predict(
    np.array([driving_record.image for driving_record in driving_records[0:split_index]], dtype=np.float32))
train_true_movement_amounts = np.array(
    [driving_record.get_movement_amount() for driving_record in driving_records[split_index:]])
test_true_movement_amounts = np.array(
    [driving_record.get_movement_amount() for driving_record in driving_records[0:split_index]])
train_σ1_list = [K.sqrt(Σ[0, 0]).numpy() for Σ in nnet.get_Σ(train_predict_results)]
train_σ2_list = [K.sqrt(Σ[1, 1]).numpy() for Σ in nnet.get_Σ(train_predict_results)]
test_σ1_list = [K.sqrt(Σ[0, 0]).numpy() for Σ in nnet.get_Σ(test_predict_results)]
test_σ2_list = [K.sqrt(Σ[1, 1]).numpy() for Σ in nnet.get_Σ(test_predict_results)]
train_y1_error_list = [abs(train_predict_results[i][0] - train_true_movement_amounts[i][0]) for i in
                       range(len(train_predict_results))]
train_y2_error_list = [abs(train_predict_results[i][1] - train_true_movement_amounts[i][1]) for i in
                       range(len(train_predict_results))]
test_y1_error_list = [abs(test_predict_results[i][0] - test_true_movement_amounts[i][0]) for i in
                      range(len(test_predict_results))]
test_y2_error_list = [abs(test_predict_results[i][1] - test_true_movement_amounts[i][1]) for i in
                      range(len(test_predict_results))]

pred_heatmap_y1_range = (min([train_predict_results[:, 0].min(), test_predict_results[:, 0].min()]),
                         max([train_predict_results[:, 0].max(), test_predict_results[:, 0].max()]))
pred_heatmap_y2_range = (min([train_predict_results[:, 1].min(), test_predict_results[:, 1].min()]),
                         max([train_predict_results[:, 1].max(), test_predict_results[:, 1].max()]))
pred_heatmap_σ1_range = (min([min(train_σ1_list), min(test_σ1_list)]),
                         max([max(train_σ1_list), max(test_σ1_list)]))
pred_heatmap_σ2_range = (min([min(train_σ2_list), min(test_σ2_list)]),
                         max([max(train_σ2_list), max(test_σ2_list)]))
y1_error_range = (min([min(train_y1_error_list), min(test_y1_error_list)]),
                  max([max(train_y1_error_list), max(test_y1_error_list)]))
y2_error_range = (min([min(train_y2_error_list), min(test_y2_error_list)]),
                  max([max(train_y2_error_list), max(test_y2_error_list)]))

output_predict_results(train_predict_results, train_true_movement_amounts, "train", pred_heatmap_y1_range,
                       pred_heatmap_y2_range, pred_heatmap_σ1_range, pred_heatmap_σ2_range,
                       y1_error_range, y2_error_range)
output_predict_results(test_predict_results, test_true_movement_amounts, "test", pred_heatmap_y1_range,
                       pred_heatmap_y2_range, pred_heatmap_σ1_range, pred_heatmap_σ2_range,
                       y1_error_range, y2_error_range)
