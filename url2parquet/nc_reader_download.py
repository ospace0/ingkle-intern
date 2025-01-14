import h5py


def nc_reader(ncfile_path, data_type):
    try:
        with h5py.File(ncfile_path, 'r') as file:
            image_data = file['image_pixel_values'][:]
            return image_data

    except Exception as e:
        print(f"{data_type} file read error error: {e}")
        return None