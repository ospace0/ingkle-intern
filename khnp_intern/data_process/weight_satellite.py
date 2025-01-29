import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def cut_satellite(satellite_cut: pd.DataFrame, generator_data: pd.DataFrame):  # 데이터 자르기
    points_in_range_list = []
    for _, gen_row in generator_data.iterrows():
        gen_lat, gen_lon = gen_row['lat'], gen_row['lon']
        points_in_range = satellite_cut[
            (satellite_cut['lat'] >= gen_lat - 0.5) & (satellite_cut['lat'] <= gen_lat + 0.5) &
            (satellite_cut['lon'] >= gen_lon - 0.5) & (satellite_cut['lon'] <= gen_lon + 0.5)]
        avg_values = points_in_range.drop(columns=['lat', 'lon', 'hour'], errors='ignore').mean()
        points_in_range_list.append(avg_values)
    total_data = pd.concat(points_in_range_list, ignore_index=True)
    return total_data

def cal_satellite(satellite_cut: pd.DataFrame, generator_data: pd.DataFrame): # 1~24시간의 가중치 평균내서 반환(3979rows)
    final_data = []
    for gen_lat, gen_lon in zip(generator_data['lat'], generator_data['lon']): # generator_Data의 각 row 확인 -> 3979번번
        points_in_range = satellite_cut[
            (satellite_cut['lat'] >= gen_lat - 0.5) & (satellite_cut['lat'] <= gen_lat + 0.5) &
            (satellite_cut['lon'] >= gen_lon - 0.5) & (satellite_cut['lon'] <= gen_lon + 0.5)].copy()
         # 범위 안에 있는 0~23시의 데이터
        distances = np.sqrt((points_in_range['lat'] - gen_lat) ** 2 + (points_in_range['lon'] - gen_lon) ** 2)
        weights = 1 / distances
        weights = weights.values.reshape(-1, 1)  # Series -> NumPy 배열 -> 2D 배열
        scaler = RobustScaler()
        robust_scaled = scaler.fit_transform(weights)
        weight_avg = robust_scaled.mean()
        final_data.append(weight_avg)
    return pd.DataFrame({'weight': final_data})

def weight_satellite(satellite_cut: pd.DataFrame, generator_data: pd.DataFrame): # 24시간 데이터 가중평균 값
    filtered_points = cut_satellite(satellite_cut, generator_data)
    result_dt = cal_satellite(satellite_cut, generator_data)
    results = filtered_points.mul(result_dt['weight'], axis=0)
    return results