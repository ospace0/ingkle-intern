import pandas as pd
from pyproj import Proj, Transformer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Lambert Conformal Conic (LCC) 투영 설정
lcc_params = {
    "lat_1": 30.0,   # 표준 평행선 1
    "lat_2": 60.0,   # 표준 평행선 2
    "lat_0": 38.0,   # 원점 위도
    "lon_0": 126.0   # 중심 자오선
}

def get_lcc_params_by_resolution(resolution):
    if resolution == 05:
        return {
            "x_start": 1430, "x_end": 2655,
            "y_start": 1545, "y_end": 2883,
            "image_width": 3600,
            "image_height": 3600,
            "x_min": -899750,
            "x_max": 899750,
            "y_min": -899750,
            "y_max": 899750,
        }
    elif resolution == 10:
        return {
            "x_start": 715, "x_end": 1327,
            "y_start": 772, "y_end": 1441,
            "image_width": 1800,
            "image_height": 1800,
            "x_min": -899500,
            "x_max": 899500,
            "y_min": -899500,
            "y_max": 899500,
        }
    elif resolution == 20:
        return {
            "x_start": 357, "x_end": 663,
            "y_start": 410, "y_end": 720,
            "image_width": 900,
            "image_height": 900,
            "x_min": -899000,
            "x_max": 899000,
            "y_min": -899000,
            "y_max": 899000,
        }

def pixel_to_latlon(args):
    pixel_x, pixel_y, params, lcc_params = args
    lcc_proj = Proj(
        proj="lcc",
        lat_1=lcc_params["lat_1"],  # 표준 평행선 1
        lat_2=lcc_params["lat_2"],  # 표준 평행선 2
        lat_0=lcc_params["lat_0"],  # 원점 위도
        lon_0=lcc_params["lon_0"],  # 중심 자오선
        x_0=0,
        y_0=0,
        ellps="WGS84"  # 타원체
    )
    wgs_proj = Proj(proj="latlong", datum="WGS84")
    transformer = Transformer.from_proj(lcc_proj, wgs_proj)
    x = params["x_min"] + (pixel_x / (params["image_width"] - 1)) * (params["x_max"] - params["x_min"])
    y = params["y_max"] - (pixel_y / (params["image_height"] - 1)) * (params["y_max"] - params["y_min"])
    lon, lat = transformer.transform(x, y)
    return pixel_x, pixel_y, lat, lon

def generate_precomputed_coordinates_parallel(resolution, output_file):
    """
    병렬 처리를 이용해 미리 계산된 좌표 데이터를 생성
    """
    params = get_lcc_params_by_resolution(resolution)

    # 좌표 목록 생성
    tasks = [
        (x, y, params, lcc_params)
        for x in range(params["x_start"], params["x_end"] + 1)
        for y in range(params["y_start"], params["y_end"] + 1)
    ]

    # 병렬 처리
    print(f"병렬 처리 시작 (해상도: {resolution})...")
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(pixel_to_latlon, tasks), total=len(tasks), desc="좌표 생성 진행"))

    # 데이터프레임 생성
    df = pd.DataFrame(results, columns=["x", "y", "Latitude", "Longitude"])

    # Parquet 파일로 저장
    df.to_parquet(output_file, index=False)
    print(f"미리 계산된 좌표 데이터가 저장되었습니다: {output_file}")


if __name__ == '__main__':
    # 해상도별 파일 생성
    resolutions = [05, 10, 20]
    base_output_dir = r"F:\INKLE\2024_01_10\precomputed_coordinates"

    for resolution in resolutions:
        output_file = f"{base_output_dir}_res_{resolution}.parquet"
        generate_precomputed_coordinates_parallel(resolution, output_file)
