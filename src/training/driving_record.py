import json
import os


class DrivingRecord:
    """
    走行記録
    """

    def __init__(self, src: dict):
        self.image_filename: str = os.path.expanduser('~/.ros/' + src['image_filename'])
        self.odometries = [self.__Odometry(odometry_src) for odometry_src in src['odometries']]

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
        with open(self.__file_path, 'r') as f:
            driving_record_json_lines: list[str] = f.readlines()

        return [DrivingRecord(json.loads(src)) for src in driving_record_json_lines]
