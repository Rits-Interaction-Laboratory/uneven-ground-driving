import logging
import sys

import numpy as np

from src.training.driving_record import DrivingRecord, DrivingRecordRepository
from src.training.nnet.base_nnet import BaseNNet
from src.training.nnet.resnet import ResNet

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
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
    logging.info(f'xの平均: {np.mean([movement_amount[0] for movement_amount in movement_amounts])}')
    logging.info(f'yの平均: {np.mean([movement_amount[1] for movement_amount in movement_amounts])}')

    covariance_matrix: np.ndarray = np.cov(movement_amounts, rowvar=False)
    for line in str(covariance_matrix).split('\n'):
        logging.info(line)


logging.info('訓練データセットをロード開始')
driving_records: list[DrivingRecord] = driving_record_repository.select_all()
output_stats(driving_records)
x: np.ndarray = np.array([driving_record.image for driving_record in driving_records], dtype=np.float32)
y: np.ndarray = np.array([driving_record.get_movement_amount() for driving_record in driving_records], dtype=np.float32)

logging.info('訓練開始')
nnet.train(x, y)
