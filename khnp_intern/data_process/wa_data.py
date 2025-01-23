import numpy as np
import pandas as pd

def weight_satellite(satellite_data: pd.DataFrame, gen_locations: pd.DataFrame) -> list:
    weighted_averages = []
    
    third_column_name = satellite_data.columns[2]

    for _, gen in gen_locations.iterrows():
        lat, lon = gen['lat'], gen['lon']

        relevant_data = satellite_data[(satellite_data['lat'] >= lat - 0.5) & (satellite_data['lat'] <= lat + 0.5) &
                                       (satellite_data['lon'] >= lon - 0.5) & (satellite_data['lon'] <= lon + 0.5)]
        
        if relevant_data.empty:
            weighted_averages.append(np.nan)
        else:
            distances = np.sqrt((relevant_data['lat'] - lat)**2 + (relevant_data['lon'] - lon)**2)
            distances = np.round(distances, 2)  # 소수점 자릿수를 2자리로 제한
            weights = 1 / (distances + 1e-6)
            weighted_sum = np.sum(weights * relevant_data[third_column_name])
            sum_of_weights = np.sum(weights)
            weighted_avg = weighted_sum / sum_of_weights
            weighted_averages.append(weighted_avg)
    
    return weighted_averages

# #brute_force 딱히 시간줄진않음;
# import numpy as np
# import pandas as pd

# def weight_satellite(satellite_data: pd.DataFrame, gen_locations: pd.DataFrame) -> list:
#     weighted_averages = []
    
#     third_column_name = satellite_data.columns[2]

#     for _, gen in gen_locations.iterrows():
#         lat, lon = gen['lat'], gen['lon']

#         relevant_data = satellite_data[(satellite_data['lat'] >= lat - 0.5) & (satellite_data['lat'] <= lat + 0.5) &
#                                        (satellite_data['lon'] >= lon - 0.5) & (satellite_data['lon'] <= lon + 0.5)]
        
#         if relevant_data.empty:
#             weighted_averages.append(np.nan)
#         else:
#             distances = np.sqrt((relevant_data['lat'] - lat)**2 + (relevant_data['lon'] - lon)**2)
#             weights = 1 / (distances + 1e-6)
#             weighted_sum = np.sum(weights * relevant_data[third_column_name])
#             sum_of_weights = np.sum(weights)
#             weighted_avg = weighted_sum / sum_of_weights
#             weighted_averages.append(weighted_avg)
    
#     return weighted_averages