import numpy as np
import pandas as pd

def weight_satellite(weights, relevant_data: pd.DataFrame, s_col) -> list:
    weighted_sum = np.sum(weights * relevant_data[s_col])
    sum_of_weights = np.sum(weights)
    weighted_avg = weighted_sum / sum_of_weights
    return weighted_avg
    

# def weight_satellite_before(satellite_data: pd.DataFrame, gen_locations: pd.DataFrame) -> pd.DataFrame:
#     weights_all_boxes = pd.DataFrame()
#     for i, gen in gen_locations.iterrows():
#         lat, lon = gen['lat'], gen['lon']

#         relevant_data = satellite_data[(satellite_data['lat'] >= lat - 0.5) & (satellite_data['lat'] <= lat + 0.5) &
#                                        (satellite_data['lon'] >= lon - 0.5) & (satellite_data['lon'] <= lon + 0.5)]
#         box = relevant_data[relevant_data['time'] == 0][['lat', 'lon']]
#         distances = np.sqrt((box['lat'] - lat)**2 + (box['lon'] - lon)**2)
#         weights = 1 / (distances + 1e-6)
#         weights_all_boxes[f'weights_gen{i}'] = weights.reset_index(drop=True)

#     return weights_all_boxes, relevant_data