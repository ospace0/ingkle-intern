import requests
import h5py
import os
import time
import numpy as np
import pandas as pd
import shutil
import stat

from datetime import datetime, timedelta
from tqdm import tqdm
from glob import glob
from io import BytesIO
from nc_read_downloaded import nc_reader
from changable_things import data_types, start_date, end_date, parquet_directory, nc_file_path, merged_directory, size_ranges, download_period, total_steps
from f_merge_parquet_files import merge_parquet_files
from f_process_and_save_parquet import process_and_save_parquet


# 메인 실행 함수
def main():
    start_time = time.time()  # 시작 시간 기록

    # 전체 날짜 진행도를 위한 tqdm
    date_progress = tqdm(total=(end_date - start_date).days, desc="전체 진행도 (Dates)")

    searching_date = start_date
    while searching_date < end_date:  # (1) 데이터 수집 전체 days 루프
        hour_progress = tqdm(total=24, desc=f"{searching_date.strftime('%Y-%m-%d')} 진행도 (Hours)", leave=False)

        # 하루의 각 시간 처리
        searching_hours = [searching_date + timedelta(hours=hour) for hour in range(24)]
        for searching_hour in searching_hours:  # (2) 하루 루프, 24시간

            time_progress = tqdm(total=total_steps, desc=f"{searching_hour.strftime('%H:%M')} 진행도", leave=False)

            searching_time = searching_hour
            hourly_data_list = []
            while searching_time < searching_hour + timedelta(hours=1):  # (3) 한 시간 루프, 주기 6개
                data_type_progress = tqdm(total=len(data_types), desc="데이터 타입 진행도", leave=False)

                file_data_per_type = []
                for data_type in data_types:  # (4) 한 주기 루프, 파장 타입 루프 16개

                    data = nc_reader(nc_file_path, data_type,searching_time)
                    file_data_per_type.append(data)  

                    data_type_progress.update(1)
                data_type_progress.close()

                hourly_data_list.append(file_data_per_type)
                searching_time += timedelta(minutes=download_period)
                time.sleep(1)

                time_progress.update(1)
            time_progress.close()

            hour_progress.update(1)

            # Parquet 파일 저장
            timestamp = searching_hour.strftime('%Y%m%d_%H')
            parquet_base_path = f"{parquet_directory}/satellite_data_1havg_{timestamp}"
            process_and_save_parquet(hourly_data_list, data_types, parquet_base_path, size_ranges)

        hour_progress.close()

        date_progress.update(1)
        searching_date += timedelta(days=1)

        merge_parquet_files(parquet_directory, merged_directory)

        # Parquet 디렉토리 내 파일 삭제
        for filename in os.listdir(parquet_directory):
            file_path = os.path.join(parquet_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('파일 삭제 중 오류 발생: %s' % e)
        print("Parquet 디렉토리 내 파일들이 성공적으로 삭제되었습니다.")


    date_progress.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"전체 과정 완료! 총 걸린 시간: {timedelta(seconds=elapsed_time)}")

if __name__ == '__main__':
    main()