import pandas as pd
from data_process.data_path import weather_path, generator_path, satellite_path

class OriginalData:
    def convert_generator(self, date_file: str):
        date_data = pd.read_csv(f"{generator_path}daily/{date_file}.csv")
        date_data["efficiency"] = date_data["generationPerHour"] / date_data["reservedCapacity"]
        date_data["genHour"] = date_data["genHour"] - 1
        date_data = date_data[date_data["genHour"] > 0]
        return date_data[["kpxGenid", "genHour", "efficiency", "lat", "lon"]]

    def convert_weather(self, date_file: str):
        date_data = pd.read_csv(f"{weather_path}daily/{date_file}.csv")
        date_data.loc[date_data["SS"].isnull(), "SI"] = 0
        col_fill_vals = {"RN": 0, "SS": 0, "SD_TOT": 0, "SD_HR3": 0}
        for col in col_fill_vals.keys():
            date_data.loc[date_data[col].isnull(), col] = col_fill_vals[col]
        date_data["time"] = pd.to_datetime(date_data["TM"]).dt.strftime("%H").astype(int)
        date_data.drop(["TM", "alt", "STN"], axis=1, inplace=True)
        return date_data
  
    def convert_satellite(self, date_file: str):
        full_path = (f"{satellite_path}{date_file}.parquet")
        date_data = pd.read_parquet(full_path)
        date_data = date_data.drop(columns=["NR013", "NR016"])
        return date_data
