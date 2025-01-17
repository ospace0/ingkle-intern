import numpy as np
import pandas as pd
from pyproj import Proj, Transformer

def _create_flat_xy(size_info: dict):
    x_vals = np.arange(size_info["x_start"], size_info["x_end"] + 1)
    y_vals = np.arange(size_info["y_start"], size_info["y_end"] + 1)
    xy_mesh = np.meshgrid(x_vals, y_vals)
    return xy_mesh[0].flatten(), xy_mesh[1].flatten()

class CoordRanger:
    def __init__(self):
        self.base_output_dir = "api/"
        self.size_info = {
            "3600": {
                "x_start": 1430, "x_end": 2655,
                "y_start": 1545, "y_end": 2883,
                "image_width": 3600,
                "image_height": 3600,
                "x_min": -899750,
                "x_max": 899750,
                "y_min": -899750,
                "y_max": 899750,
            },
            "1800": {
                "x_start": 715, "x_end": 1327,
                "y_start": 772, "y_end": 1441,
                "image_width": 1800,
                "image_height": 1800,
                "x_min": -899500,
                "x_max": 899500,
                "y_min": -899500,
                "y_max": 899500,
            },
            "900": {
                "x_start": 357, "x_end": 663,
                "y_start": 410, "y_end": 720,
                "image_width": 900,
                "image_height": 900,
                "x_min": -899000,
                "x_max": 899000,
                "y_min": -899000,
                "y_max": 899000,
            }
        }
        self.lcc_params = {
            "lat_1": 30.0,   # 표준 평행선 1
            "lat_2": 60.0,   # 표준 평행선 2
            "lat_0": 38.0,   # 원점 위도
            "lon_0": 126.0   # 중심 자오선
        }

    def pixel_to_latlon(self, size: str):
        output_file = f"{self.base_output_dir}coordinate_size{size}.parquet"
        size_info = self.size_info[size]
        pixel_x, pixel_y = _create_flat_xy(size_info)
        lcc_proj = Proj(
            proj="lcc",
            lat_1=self.lcc_params["lat_1"],  # 표준 평행선 1
            lat_2=self.lcc_params["lat_2"],  # 표준 평행선 2
            lat_0=self.lcc_params["lat_0"],  # 원점 위도
            lon_0=self.lcc_params["lon_0"],  # 중심 자오선
            x_0=0,
            y_0=0,
            ellps="WGS84"  # 타원체
        )
        wgs_proj = Proj(proj="latlong", datum="WGS84")
        transformer = Transformer.from_proj(lcc_proj, wgs_proj)
        x = size_info["x_min"] + (pixel_x / (size_info["image_width"] - 1)) * (size_info["x_max"] - size_info["x_min"])
        y = size_info["y_max"] - (pixel_y / (size_info["image_height"] - 1)) * (size_info["y_max"] - size_info["y_min"])
        lon, lat = transformer.transform(x, y)
        df = pd.DataFrame(np.vstack([pixel_x, pixel_y, lat, lon]).T, columns=["x", "y", "Latitude", "Longitude"])
        df[["x", "y"]] = df[["x", "y"]] - df[["x", "y"]].min(axis=0)

        # Parquet 파일로 저장
        df.to_parquet(output_file, index=False)

if __name__ == '__main__':
    # 해상도별 파일 생성
    sizes = ["900", "1800", "3600"]
    ranger = CoordRanger()
    for s in sizes:
        ranger.pixel_to_latlon(s)
