import requests
import h5py
import os
from io import BytesIO
from tqdm import tqdm
from changable_things import auth_key, region, nc_file_path

def nc_reader_downloaded(ncfile_path, data_type, searching_time):
    searching_time_name = searching_time.strftime("%Y%m%d%H%M")
    path = os.path.join(ncfile_path, f"{data_type}_{searching_time_name}.nc").replace("\\", "/")
    try:
        with h5py.File(path, 'r') as file:
            image_data = file['image_pixel_values'][:]
            return image_data

    except Exception as e:
        print(f"{data_type} file read error error: {e}")
        return None
    
def nc_reader_url(url, data_type):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with h5py.File(BytesIO(response.content), 'r') as file:
            image_data = file['image_pixel_values'][:]
            return image_data

    except Exception as e:
        print(f"{data_type} 다운로드 오류: {e}")
        return None
    
def process_data_types(get_nc, data_types, searching_time):
    data_type_progress = tqdm(total=len(data_types), desc="데이터 타입 진행도", leave=False)

    file_data_per_type = []
    for data_type in data_types:  # (4) one data type loop

        if get_nc == "url":
            url = f"https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/{data_type}/{region}/data?date={searching_time.strftime('%Y%m%d%H%M')}&authKey={auth_key}"
            data = nc_reader_url(url, data_type)
        else:
            data = nc_reader_downloaded(nc_file_path, data_type, searching_time)
            
        file_data_per_type.append(data)                                

        data_type_progress.update(1)
    data_type_progress.close()
    return file_data_per_type