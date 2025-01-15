import h5py
import os
from tqdm import tqdm

def process_data_types(nc_file_path, data_types, searching_time):
    data_type_progress = tqdm(total=len(data_types), desc="데이터 타입 진행도", leave=False)

    file_data_per_type = []
    for data_type in data_types:  # (4) 한 주기 루프, 파장 타입 루프 16개

        data = nc_reader(nc_file_path, data_type,searching_time)
        file_data_per_type.append(data)  

        data_type_progress.update(1)
    data_type_progress.close()
    return file_data_per_type

def nc_reader(ncfile_path, data_type, searching_time):
    searching_time_name = searching_time.strftime("%Y%m%d%H%M")
    path = os.path.join(ncfile_path, f"{data_type}_{searching_time_name}.nc").replace("\\", "/")
    try:
        with h5py.File(path, 'r') as file:
            image_data = file['image_pixel_values'][:]
            return image_data

    except Exception as e:
        print(f"{data_type} file read error error: {e}")
        return None