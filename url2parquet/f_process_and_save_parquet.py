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


# 이미지 데이터를 Parquet로 저장하는 함수
def process_and_save_parquet(data_list, data_types, parquet_save_path, size_ranges):
    if parquet_directory:
        if not os.path.exists(parquet_directory):
            os.makedirs(parquet_directory)
            
    if not data_list:
        print("수집된 데이터가 없습니다.")
        return

    size_grouped_data = {}
    for data_type_index, data_type in enumerate(data_types):
        all_data_for_type = [
            file_data[data_type_index] for file_data in data_list if file_data is not None and file_data[data_type_index] is not None
        ]

        if not all_data_for_type:
            print(f"{data_type}에 대한 유효한 데이터가 없습니다.")
            continue

        # 데이터 크기와 범위 선택
        rows, cols = all_data_for_type[0].shape
        if (rows, cols) in size_ranges:
            x_range, y_range = size_ranges[(rows, cols)]
        else:
            print(f"{rows}x{cols} 크기의 데이터는 범위 정의가 없습니다. 스킵합니다.")
            continue

        # 필요한 범위 자르기
        all_data_for_type = [data[y_range[0]:y_range[1] +1, x_range[0]:x_range[1] +1] for data in all_data_for_type]

        # 평균 계산
        averaged_data = np.nanmean(np.stack(all_data_for_type), axis=0)

        # 소수점 자리수 제한
        averaged_data = np.round(averaged_data, decimals=3)

        # float16으로 변환
        averaged_data = averaged_data.astype(np.float16)

        # 좌표 생성 및 결과 데이터 저장
        rows, cols = averaged_data.shape  #cut data의 행의 개수, 열의 개수 -> 행의 개수면 y좌표수랑 같겠죠..
        x, y = np.meshgrid(range(cols), range(rows))
        flattened_data = pd.DataFrame({
            'x': x.ravel(),
            'y': y.ravel(),
            data_type: averaged_data.ravel()
        })

        size_cut = f"{cols}x{rows}"
        if size_cut not in size_grouped_data:
            size_grouped_data[size_cut] = flattened_data
        else:
            size_grouped_data[size_cut] = pd.merge(size_grouped_data[size_cut], flattened_data, on=['x', 'y'], how='outer')

    # Parquet 저장
    for size_cut, combined_data in size_grouped_data.items():
        parquet_save_path_with_size = f"{parquet_save_path}_{size_cut}.parquet"
        combined_data.to_parquet(parquet_save_path_with_size, engine='pyarrow', compression='snappy')
        print(f"\nParquet 파일이 {parquet_save_path_with_size}에 저장되었습니다.")