# import numpy as np
# import pandas as pd

# def weight_satellite(satellite_data: pd.DataFrame, gen_locations: pd.DataFrame) -> list:
#     weighted_averages = []
    
#     for _, gen in gen_locations.iterrows():
#         lat, lon = gen['lat'], gen['lon']

#         relevant_data: pd.DataFrame = satellite_data[(satellite_data['lat'] >= lat - 0.5) & (satellite_data['lat'] <= lat + 0.5) &
#                                        (satellite_data['lon'] >= lon - 0.5) & (satellite_data['lon'] <= lon + 0.5)]
        
#         if relevant_data.empty:
#             weighted_averages.append(np.nan)
#         else:
#             distances = np.sqrt((relevant_data['lat'] - lat)**2 + (relevant_data['lon'] - lon)**2)
#             weights = 1 / (distances + 1e-6)  
#             weighted_avg = np.average(relevant_data.iloc[:, 2], weights=weights)  
#             weighted_averages.append(weighted_avg)
    
#     return weighted_averages


import pandas as pd
import numpy as np
from data_process.original_data import OriginalData
from data_process.data_path import satellite_path

def cut_satellite(satellite_cut: pd.DataFrame, generator_data: pd.DataFrame):
    filtered_points = []
    point_dt = satellite_cut.copy()
    point_dt.rename(columns={'Latitude': 'Point Latitude', 'Longitude': 'Point Longitude'}, inplace=True)
    for _, gen in generator_data.iterrows():
        gen_lat = gen['Latitude']
        gen_lon = gen['Longitude']
        gen_id = gen['kpxGenid']
        mask = ((point_dt['Point Latitude'] >= gen_lat - 0.5) &
            (point_dt['Point Latitude'] <= gen_lat + 0.5) &
            (point_dt['Point Longitude'] >= gen_lon - 0.5) &
            (point_dt['Point Longitude'] <= gen_lon + 0.5))
        filtered = point_dt[mask].copy()
        if not filtered.empty:
            filtered['Generator ID'] = gen_id
            filtered['Generator Latitude'] = gen_lat
            filtered['Generator Longitude'] = gen_lon
            filtered_points.append(filtered)
    return pd.concat(filtered_points, ignore_index=True)
def calculate_weights(generator_df: pd.DataFrame, filtered_points: pd.DataFrame):
    results = []
    data_columns = [col for col in filtered_points.columns if col not in ['Generator ID', 'Point Latitude', 'Point Longitude', 'Generator Latitude', 'Generator Longitude']]
    for _, gen in generator_df.iterrows():
        gen_lat = gen['Latitude']
        gen_lon = gen['Longitude']
        gen_id = gen['kpxGenid']
        gen_points = filtered_points[filtered_points['Generator ID'] == gen_id]
        distances = np.sqrt((gen_points['Point Latitude'] - gen_lat) ** 2 +
                            (gen_points['Point Longitude'] - gen_lon) ** 2)
        one_over_l = 1 / (distances + 1e-9)
        weighted_values = {}
        for col in data_columns:
            weighted_values[f'Weighted {col}'] = np.sum(one_over_l * gen_points[col]) / np.sum(one_over_l)
        result = {
            'Generator ID': gen_id,
            'Generator Latitude': gen_lat,
            'Generator Longitude': gen_lon,
        }
        result.update(weighted_values)
        results.append(result)
    return pd.DataFrame(results)
def weight_satellite(generator_data: pd.DataFrame, satellite_cut: pd.DataFrame):
    filtered_points = cut_satellite(satellite_cut, generator_data)
    result_df = calculate_weights(generator_data, filtered_points)
    return result_df