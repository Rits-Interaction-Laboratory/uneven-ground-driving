import json
import os

import numpy as np
import tensorflow as tf
import tqdm
from keras_preprocessing.image import img_to_array, load_img


class DrivingRecord:
    """
    走行記録
    """

    def __init__(self, src: dict):
        self.image_filename: str = os.path.expanduser('~/.ros/' + src['image_filename'])
        self.image: np.ndarray = tf.image.resize(img_to_array(load_img(self.image_filename, color_mode="grayscale")),
                                                 (64, 64))
        self.odometries = [self.__Odometry(odometry_src) for odometry_src in src['odometries']]

    def get_movement_amount(self) -> tuple[float, float]:
        """
        移動量を取得

        :return: [xの移動量, yの移動量]
        """

        return (
            self.odometries[-1].position.x - self.odometries[0].position.x,
            self.odometries[-1].position.y - self.odometries[0].position.y,
        )

    class __Odometry:
        """
        ロボット位置
        """

        def __init__(self, src: dict):
            self.position = self.__Position(src['position'])
            self.orientation = self.__Orientation(src['orientation'])

        class __Position:
            def __init__(self, src: dict):
                self.x: float = src['x']
                self.y: float = src['y']
                self.z: float = src['z']

            def __str__(self) -> str:
                return json.dumps(self.__dict__)

        class __Orientation:
            def __init__(self, src: dict):
                self.x: float = src['x']
                self.y: float = src['y']
                self.z: float = src['z']
                self.w: float = src['w']

            def __str__(self) -> str:
                return json.dumps(self.__dict__)


class DrivingRecordRepository:
    """
    走行記録リポジトリ
    """

    def __init__(self):
        self.__file_path: str = os.path.expanduser('~/.ros/uneven-ground-driving-result/result.jsonl')

    def select_all(self) -> list[DrivingRecord]:
        """
        走行記録を全件取得
        """

        with open(self.__file_path, 'r') as f:
            driving_record_json_lines: list[str] = f.readlines()

        driving_records: list[DrivingRecord] = []
        for src in tqdm.tqdm(driving_record_json_lines):
            try:
                driving_records.append(DrivingRecord(json.loads(src)))
            except (FileNotFoundError, ImportError, ValueError, KeyError):
                pass

        return driving_records
