import time

from config import Config
from odometry import OdometryModel, OdometryRepository


class ContextHolder:
    """
    Application Context Holder
    """

    def __init__(self):
        self.__timestamp: float = 0.0
        self.__odometry_repository = OdometryRepository()

    def start_to_measure(self):
        """
        計測開始時に保有したデータを初期化する
        """

        self.__timestamp: float = time.time()
        self.__odometry_repository.delete_all()

    def add_odometry_history(self, odometry: OdometryModel):
        """
        ロボット位置の履歴を追加する
        """

        self.__odometry_repository.insert(odometry)

    def get_odometry_histories(self) -> list[OdometryModel]:
        """
        ロボット位置の履歴リストを取得
        """

        return self.__odometry_repository.select_all()

    @property
    def timestamp(self) -> float:
        return self.__timestamp

    @property
    def image_filename(self) -> str:
        return f'{Config.IMAGES_PATH}/{self.timestamp}.png'

    @property
    def image_npy_filename(self) -> str:
        return f'{Config.IMAGE_NPY_PATH}/{self.timestamp}.npy'
