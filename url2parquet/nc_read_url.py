import requests
import h5py
from io import BytesIO

def nc_reader(url, data_type):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with h5py.File(BytesIO(response.content), 'r') as file:
            image_data = file['image_pixel_values'][:]
            return image_data

    except Exception as e:
        print(f"{data_type} 다운로드 오류: {e}")
        return None