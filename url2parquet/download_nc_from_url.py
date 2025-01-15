import os
from datetime import datetime, timedelta
import requests
import h5py
from tqdm import tqdm
from changable_things import auth_key, region, nc_file_path, download_period, start_date, end_date, data_types


if not os.path.exists(nc_file_path):
        os.makedirs(nc_file_path)

def download_nc_file(url, save_path):
    """Download .nc file from the given URL and save it to save_path."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"파일 저장 성공: {save_path}")
    except Exception as e:
        print(f"다운로드 실패 {url}: {e}")

def download_data(start_date, end_date, data_types, download_period):
    """Download data for all data types and periods."""
    searching_time = start_date

    while searching_time < end_date:
        for data_type in data_types:
            url = (f"https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/{data_type}/{region}/data?"
                   f"date={searching_time.strftime('%Y%m%d%H%M')}&authKey={auth_key}")
            save_file_name = f"{data_type}_{searching_time.strftime('%Y%m%d%H%M')}.nc"
            save_path = os.path.join(nc_file_path, save_file_name)
            
            if not os.path.exists(save_path):  # 중복 다운로드 방지
                download_nc_file(url, save_path)
        
        searching_time += timedelta(minutes=download_period)

# 실행
download_data(start_date, end_date, data_types, download_period)
