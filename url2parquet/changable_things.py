from datetime import datetime, timedelta

data_types = ["VI004", "VI005", "VI006", "VI008", "NR013", "NR016", "SW038", "WV063", "WV069", "WV073", "IR087", "IR096", "IR105", "IR112", "IR123", "IR133"]
start_date = datetime(2025, 1, 8)  # 검색 시작일
end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
parquet_directory = "D:/sat_file/temp_parquet_hourly"  # 임시 시간별 parquet파일 저장 경로
nc_file_path = "D:/sat_file/temp"  # nc file path
merged_directory = "D:/sat_file/daily_parquets"  # 일별 parquet files

# 필요한 크기와 범위 정의
size_ranges = {
        (3600, 3600): [(1545, 2883), (1430, 2655)], 
        (1800, 1800): [(772, 1441), (715, 1327)],  
        (900, 900): [(410, 720), (357, 663)],  
    }

download_period = 10  # 다운로드 주기
total_steps = 60 // download_period  # 다운로드 주기에 따른 단계 수