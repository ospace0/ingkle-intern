from datetime import datetime, timedelta

auth_key = "6vdMscAHSSC3TLHABykgvw"
data_types = ["VI004", "VI005", "VI006", "VI008", "NR013", "NR016", "SW038", "WV063", "WV069", "WV073", "IR087", "IR096", "IR105", "IR112", "IR123", "IR133"]
region = "KO"
start_date = datetime(2025, 1, 10)  # 검색 시작일
end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
parquet_directory = "D:/sat_file/temp_parquet_hourly"  # 임시 시간별 parquet파일 저장 경로
merged_directory = "D:/sat_file/daily_parquets"  # 일별 parquet files