import requests, h5py, time
from datetime import datetime, timedelta
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _judge_api_time():
    time_hour = datetime.now().hour
    judge = (time_hour >= 16 and time_hour <=23)
    return judge

def _nc_reader_url(url, data_type: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with h5py.File(BytesIO(response.content), 'r') as file:
            image_data = file['image_pixel_values'][:]
            return image_data

    except Exception as e:
        print(f"{data_type} 다운로드 오류: {e}")
        return None

def _data_xy_add(data_type: str, dataset: list):
    averaged_data = np.round(np.nanmean(dataset, axis=0), 3).astype(np.int16)
    rows, cols = averaged_data.shape  #cut data의 행의 개수, 열의 개수 -> 행의 개수면 y좌표수랑 같겠죠..
    x, y = np.meshgrid(range(cols), range(rows))
    flattened_data = pd.DataFrame({
        'x': x.ravel(),
        'y': y.ravel(),
        data_type: averaged_data.ravel()
    })
    return flattened_data

class SatelliteData:
    def __init__(self):
        self.big_api_use = _judge_api_time()
        self.data_types = ["VI004", "VI005", "VI006", "VI008", "NR013", "NR016", 
                           "SW038", "WV063", "WV069", "WV073", "IR087", "IR096", 
                           "IR105", "IR112", "IR123", "IR133"]
        self.types_size = {
            "3600": ["VI006"],
            "1800": ["VI004", "VI005", "VI008"],
            "900": ["NR013", "NR016", "SW038", "WV063", "WV069", "WV073", "IR087", 
                  "IR096", "IR105", "IR112", "IR123", "IR133"]
        }
        self.size_ranges = {
            (3600, 3600): [(1545, 2883), (1430, 2655)], 
            (1800, 1800): [(772, 1441), (715, 1327)],  
            (900, 900): [(410, 720), (357, 663)],  
        }
        if self.big_api_use:
            self.auth_key = "-kf5v_egQ6-H-b_3oAOv7A"
        else:
            self.auth_key = "WyoyNzB6RcWqMjcwekXFMA"
        self.region = "KO"
        self.download_timedelta = timedelta(minutes=60)

    def _extract_data(self, data: np.ndarray):
        x_range, y_range = self.size_ranges[data.shape]
        range_data = data[x_range[0]:x_range[1] +1, y_range[0]:y_range[1] +1]
        # plt.figure(figsize=(12, 8))
        # plt.subplot(1,2,1)
        # plt.imshow(data)
        # plt.subplot(1,2,2)
        # plt.imshow(range_data)
        # plt.show()
        return range_data

    def _download_types_data(self, save_time: datetime):
        save_time_str = save_time.strftime('%Y%m%d%H%M')
        type_dataset = {}
        for data_type in self.data_types:
            if self.big_api_use:
                url = f"https://apihub-org.kma.go.kr/api/typ05/api/GK2A/LE1B/{data_type}/{self.region}/data?date={save_time_str}&authKey={self.auth_key}"
            else:
                url = f"https://apihub.kma.go.kr/api/typ05/api/GK2A/LE1B/{data_type}/{self.region}/data?date={save_time_str}&authKey={self.auth_key}"
            data = _nc_reader_url(url, data_type)
            data = self._extract_data(data)
            type_dataset[data_type] = data
        return type_dataset

    def _convert_type(self, dataset: pd.DataFrame):
        dataset[["lat", "lon"]] = dataset[["lat", "lon"]].astype(np.float32)
        dataset[["hour"]] = dataset[["hour"]].astype(np.int8)
        return dataset

    def _convert_coord(self, size: str, dataset: pd.DataFrame):
        coord_file = f"api/coordinate_size{size}.parquet"
        coord_df = pd.read_parquet(coord_file)
        if np.abs(dataset[['x', 'y']] - coord_df[['x', 'y']]).sum().max() > 0:
            raise Exception("coordinate information incorrect")
        dataset[["lat", "lon"]] = coord_df[["Latitude", "Longitude"]]
        dataset = dataset.drop(columns=["x", "y"])
        dataset = self._convert_type(dataset)
        return dataset

    def _save_date_data(self, save_date: datetime):
        date_dataset = {k:[] for k in self.types_size.keys()}
        for save_hour in range(24):
            hourly_dataset = {k:[] for k in self.types_size.keys()}
            save_timehour = save_date.replace(hour=save_hour)
            save_timeminute = save_timehour
            in_hour_dataset = {k:[] for k in self.data_types}
            while save_timeminute < save_timehour + timedelta(hours=1):
                types_dataset = self._download_types_data(save_timeminute)
                for k in types_dataset.keys():
                    in_hour_dataset[k].append(types_dataset[k])
                save_timeminute += self.download_timedelta
            in_hour_dataset = {k:_data_xy_add(k, in_hour_dataset[k]) for k in types_dataset.keys()}
            for size_type in self.types_size.keys():
                for data_type in self.data_types:
                    if data_type in self.types_size[size_type]:
                        if len(hourly_dataset[size_type]) < 1:
                            hourly_dataset[size_type] = in_hour_dataset[data_type]
                        else:
                            hourly_dataset[size_type] = pd.merge(hourly_dataset[size_type], in_hour_dataset[data_type], on=['x', 'y'], how='outer')
                        continue
            for k in hourly_dataset.keys():
                if len(hourly_dataset[k]) < 1:
                    continue
                hourly_dataset[k]["hour"] = save_hour
                hourly_dataset[k]["datetime"] = np.int32(int(save_date.strftime("%Y%m%d")))
                date_dataset[k].append(hourly_dataset[k])
        for k in date_dataset.keys():
            if len(date_dataset[k]) < 1:
                continue
            date_str = save_date.strftime("%Y-%m-%d")
            save_path = f"D:/khnp_solar_power/satellite/daily/size{k}/date {date_str} size{k} data.parquet"
            date_size_data = pd.concat(date_dataset[k])
            date_size_data = self._convert_coord(k, date_size_data)
            date_size_data = self._convert_type(date_size_data)
            date_size_data.to_parquet(save_path)

    def save_all_data(self, start_date: datetime, end_date: datetime):
        save_date = start_date
        while save_date < end_date:
            date_data_name = save_date.strftime("%Y-%m-%d")
            save_file_name = f"date {date_data_name} data.csv"
            # if save_file_name in os.listdir("khnp_intern/sample_data/satellite/daily/"):
            #     save_date += timedelta(1)
            #     continue
            self._save_date_data(save_date)
            print(save_date)
            save_date += timedelta(1)

if __name__ == "__main__":
    start_date = datetime(2025,1,10)
    end_date = datetime.today().replace(hour=0, minute=0, second=0)

    satellite_data = SatelliteData()
    satellite_data.save_all_data(start_date, end_date)
