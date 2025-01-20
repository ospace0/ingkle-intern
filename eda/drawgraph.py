import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Load the shapefile
shapefile_path = "khnp_intern/kr_shp/kr.shp"
gdf = gpd.read_file(shapefile_path)

input_path = "C:/khnp_solar_power/satelite/final_combined_with_coordinates_1.0.parquet" #change file path
input_data = pd.read_parquet(input_path)

input_data['geometry'] = input_data.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
input_gdf = gpd.GeoDataFrame(input_data, geometry='geometry', crs="EPSG:4326")

# Get bounds of the shapefile
minx, miny, maxx, maxy = gdf.total_bounds

# Define the spacing of the grid
grid_spacing = 0.05

# Create grid points
x_coords = np.arange(minx, maxx, grid_spacing)
y_coords = np.arange(miny, maxy, grid_spacing)
wide_area = np.vstack([coord.flatten() for coord in np.meshgrid(x_coords, y_coords)]).T
points = [Point(x, y) for x in x_coords for y in y_coords]

# Convert to GeoDataFrame
grid_gdf = gpd.GeoDataFrame(geometry=points, crs=gdf.crs)

# Clip the grid points using the shapefile geometry
clipped_grid_gdf = gpd.clip(grid_gdf, gdf)

grid_coords = np.array([(point.x, point.y) for point in clipped_grid_gdf.geometry])

joined_gdf = gpd.sjoin_nearest(clipped_grid_gdf, input_gdf, how="left")

# Extract IR087 values for the grid points
grid_coords = np.array([(point.x, point.y) for point in joined_gdf.geometry])
VI005_values = joined_gdf['VI005'].fillna(0).values #change values


print(grid_coords.shape)
plt.figure(figsize=(12, 10))
plt.scatter(wide_area[:,0], wide_area[:,1], s=3, color="gray", alpha=0.2, label="Korea")
plt.scatter(grid_coords[:, 0], grid_coords[:, 1], c=VI005_values, cmap="viridis", s=10, alpha=0.7, label="VI005 Data")
plt.colorbar(label="VI005 Value")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.title("Scatter Plot with IR087 Values")
plt.grid(True)
plt.show()