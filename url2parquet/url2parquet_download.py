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
from nc_reader_download import nc_reader


# 이미지 데이터를 Parquet로 저장하는 함수
def process_and_save_parquet(data_list, data_types, parquet_save_path, size_ranges):
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

# 메인 실행 함수
def main():
    start_time = time.time()  # 시작 시간 기록

    data_types = ["VI004", "VI005", "VI006", "VI008", "NR013", "NR016", "SW038", "WV063", "WV069", "WV073", "IR087", "IR096", "IR105", "IR112", "IR123", "IR133"]
    region = "KO"
    start_date = datetime(2025, 1, 10)  # 검색 시작일
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    parquet_directory = "D:/sat_file/temp_parquet_hourly"  # 임시 시간별 parquet파일 저장 경로
    nc_file_path = "D:/sat_file/temp"  # nc file path
    merged_directory = "D:/sat_file/daily_parquets"  # 일별 parquet files
    os.makedirs(parquet_directory, exist_ok=True)

    # 필요한 크기와 범위 정의
    size_ranges = {
        (3600, 3600): [(1545, 2883), (1430, 2655)], 
        (1800, 1800): [(772, 1441), (715, 1327)],  
        (900, 900): [(410, 720), (357, 663)],  
    }

    # 전체 날짜 진행도를 위한 tqdm
    date_progress = tqdm(total=(end_date - start_date).days, desc="전체 진행도 (Dates)")

    searching_date = start_date
    while searching_date < end_date:  # (1) 데이터 수집 전체 days 루프
        hour_progress = tqdm(total=24, desc=f"{searching_date.strftime('%Y-%m-%d')} 진행도 (Hours)", leave=False)

        # 하루의 각 시간 처리
        searching_hours = [searching_date + timedelta(hours=hour) for hour in range(24)]
        for searching_hour in searching_hours:  # (2) 하루 루프, 24시간
            download_period = 10  # 다운로드 주기
            total_steps = 60 // download_period  # 다운로드 주기에 따른 단계 수
            time_progress = tqdm(total=total_steps, desc=f"{searching_hour.strftime('%H:%M')} 진행도", leave=False)

            searching_time = searching_hour
            hourly_data_list = []
            while searching_time < searching_hour + timedelta(hours=1):  # (3) 한 시간 루프, 주기 6개
                data_type_progress = tqdm(total=len(data_types), desc="데이터 타입 진행도", leave=False)

                file_data_per_type = []
                for data_type in data_types:  # (4) 한 주기 루프, 파장 타입 루프 16개

                    data = nc_reader(nc_file_path, data_type)
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