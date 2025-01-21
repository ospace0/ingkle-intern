import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def weight_satellite(satellite_data: pd.DataFrame, gen_locations: pd.DataFrame) -> list:
    weighted_averages = []
    
    tree = cKDTree(satellite_data[['lat', 'lon']])
    
    for _, gen in gen_locations.iterrows():
        lat, lon = gen['lat'], gen['lon']
    
        indices = tree.query_ball_point([lat, lon], r=0.5)
        
        if not indices:
            weighted_averages.append(np.nan)
        else:
            relevant_data = satellite_data.iloc[indices]
            distances = np.sqrt((relevant_data['lat'] - lat)**2 + (relevant_data['lon'] - lon)**2)
            weights = 1 / (distances + 1e-6) 
            weighted_avg = np.average(relevant_data.iloc[:, 2], weights=weights)
            weighted_averages.append(weighted_avg)
    
    return weighted_averages