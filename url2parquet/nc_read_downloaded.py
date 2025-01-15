import h5py
import os

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