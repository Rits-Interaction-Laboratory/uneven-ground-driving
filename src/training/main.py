from driving_record import DrivingRecord, DrivingRecordRepository

# とりあえず、訓練スクリプトのみ定義する
driving_record_repository: DrivingRecordRepository = DrivingRecordRepository()
driving_records: list[DrivingRecord] = driving_record_repository.select_all()

print(driving_records[0].image_filename)
