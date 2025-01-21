import os
import pandas as pd
from data_process.data_path import weather_path, satellite_path, generator_path

class OriginalData:
    def convert_generator(self, date_file: str):
        date_data = pd.read_csv(f"{generator_path}daily/{date_file}")
        date_data["efficiency"] = date_data["generationPerHour"] / date_data["reservedCapacity"]
        return date_data[["kpxGenid", "genHour", "efficiency", "lat", "lon"]]

    def convert_weather(self, date_file: str):
        date_data = pd.read_csv(f"{weather_path}daily/{date_file}")
        date_data.loc[date_data["SS"].isnull(), "SI"] = 0
        col_fill_vals = {"RN": 0, "SS": 0, "SD_TOT": 0, "SD_HR3": 0}
        for col in col_fill_vals.keys():
            date_data.loc[date_data[col].isnull(), col] = col_fill_vals[col]
        date_data["time"] = pd.to_datetime(date_data["TM"]).dt.strftime("%H").astype(int)
        date_data.drop(["TM", "alt", "STN"], axis=1, inplace=True)
        return date_data

    def convert_satellite(self, date_file: str):
        sizes = [900, 1800, 3600]
        base_file_name = " ".join(date_file.split()[:2])  # 'date 2025-01-10' 부분만 추출
        all_data = {}

        for size in sizes:
            file_path = f"{satellite_path}daily/size{size}/{base_file_name} size{size} data.parquet"
            if os.path.exists(file_path):
                date_data = pd.read_parquet(file_path)
                date_data["time"] = date_data["hour"].astype(int)
                date_data.drop(["hour", "datetime"], axis=1, inplace=True)
                date_data = date_data[["lat", "lon", "time"] + [col for col in date_data.columns if col not in ["lat", "lon", "time"]]]
                all_data[size] = date_data

        if not all_data:
            raise FileNotFoundError(f"No file found for {date_file} in sizes {sizes}")

        return all_data