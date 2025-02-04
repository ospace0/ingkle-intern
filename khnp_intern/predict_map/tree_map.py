import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process.data_path import estimate_result_path

def _plot_efficiency(dataset: pd.DataFrame, file_name: str, 
                     v_min: float = None, v_max: float = None):
    eff_average = dataset[["lon", "lat", "efficiency"]].groupby(["lon", "lat"]).mean()
    lon = eff_average.index.get_level_values("lon")
    lat = eff_average.index.get_level_values("lat")
    eff = np.array(eff_average)
    plt.scatter(lon, lat, s=3, c=eff, cmap="rainbow", vmin=v_min, vmax=v_max)
    plt.colorbar()
    plt.savefig(f"{estimate_result_path}{model_name}/figures/{file_name}.png")
    plt.close()

if __name__ == "__main__":
    model_name = "GBR"
    data_dir = f"{estimate_result_path}{model_name}/"
    date_files = sorted([f for f in os.listdir(data_dir) if f.split(".")[-1]=="parquet"])
    all_date_data, day_date_data = [], []
    for date_f in date_files:
        date_df = pd.read_parquet(f"{data_dir}{date_f}")
        plot_data = date_df[["genHour", "lon", "lat", "efficiency"]].copy()
        plot_data["date"] = datetime.strptime(date_f, "date %Y-%m-%d data.parquet")
        all_date_data.append(plot_data.copy())
        day_date_data.append(plot_data[(plot_data["genHour"]>=10) & (plot_data["genHour"]<=16)].copy())
    all_date_data = pd.concat(all_date_data)
    day_date_data = pd.concat(day_date_data)
    _plot_efficiency(all_date_data, "whole_efficiency")
    _plot_efficiency(day_date_data, "day_efficiency")
