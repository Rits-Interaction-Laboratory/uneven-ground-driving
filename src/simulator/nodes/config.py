class Config:
    BASE_RESULT_PATH: str = 'uneven-ground-driving-result'
    """
    結果を保存するパス（~/.ros直下に保存される）
    """

    IMAGES_PATH: str = f'{BASE_RESULT_PATH}/images'
    """
    画像を保存するパス
    """

    MEASURED_RESULT_FILENAME: str = f'{BASE_RESULT_PATH}/result.jsonl'
    """
    測定結果を保存するファイル名
    """

    TURTLEBOT3_MODEL_NAME: str = 'turtlebot3_waffle'
    """
    TurtleBot3のモデル名
    """

    PERIOD_TO_MOVE: float = 3.0
    """
    ロボットを前進させる期間 [秒]
    """

    LINEAR_VELOCITY: float = 0.26
    """
    移動速度速度

    waffle: 0.0 ~ 0.26
    burger: 0.0 ~ 0.22
    """

    IS_DRY_RUN: bool = False
    """
    DRY RUNモードで実行するか
    """
