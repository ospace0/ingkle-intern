import requests
import h5py
import os
import time
import numpy as np
import pandas as pd
import shutil
from datetime import datetime, timedelta
from tqdm import tqdm
from glob import glob
from io import BytesIO
from nc_read_downloaded import nc_reader
from changable_things import data_types, start_date, end_date, parquet_directory, nc_file_path, merged_directory, size_ranges, download_period, total_steps
import stat

# Parquet 파일 병합 함수
def merge_parquet_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Parquet 파일 검색
    parquet_files = glob(os.path.join(input_directory, "*.parquet"))

    # 파일 정보 추출 및 그룹화
    grouped_files = {}
    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        parts = file_name.split("_")

        # 날짜, 시간, 배열 크기 추출
        date = parts[3]
        time = parts[4]
        size = parts[5].replace(".parquet", "")

        # 그룹 키 생성 (날짜 + 배열 크기)
        group_key = f"{date}_{size}"

        if group_key not in grouped_files:
            grouped_files[group_key] = []
        grouped_files[group_key].append((time, file_path))

    # 그룹별 파일 병합
    for group_key, files in grouped_files.items():
        files.sort()  # 시간 순서대로 정렬

        merged_data = []
        for time, file_path in files:
            df = pd.read_parquet(file_path)  # Parquet 파일 읽기

            # 시간 컬럼 추가 (yyyymmddhh)
            hour = time.zfill(2)  # 시간 형식을 두 자리로 맞춤
            datetime_str = group_key.split("_")[0] + hour
            df.insert(0, "Datetime", datetime_str)

            merged_data.append(df)

        # 병합된 데이터 저장
        merged_df = pd.concat(merged_data, ignore_index=True)
        output_file = os.path.join(output_directory, f"merged_{group_key}.parquet")
        merged_df.to_parquet(output_file, index=False)  # Parquet 파일로 저장

    print("모든 Parquet 파일이 병합되었습니다!")