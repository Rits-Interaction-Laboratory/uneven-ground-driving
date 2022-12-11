class Config:
    MEASURED_RESULT_FILENAME: str = "data.jsonl"
    """
    測定結果を保存するファイル名(~/.rosに保存される)
    """

    TURTLEBOT3_MODEL_NAME: str = "turtlebot3_waffle"
    """
    TurtleBot3のモデル名
    """

    PERIOD_TO_MOVE: float = 3.0
    """
    ロボットを前進させる期間 [秒]
    """

    MAX_VELOCITY: float = 0.26
    """
    最大速度

    waffle: 0.26
    burger: 0.22
    """
