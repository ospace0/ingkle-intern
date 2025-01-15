import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Load the shapefile
shapefile_path = "khnp_intern/kr_shp/kr.shp"
gdf = gpd.read_file(shapefile_path)

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

print(grid_coords.shape)
plt.scatter(wide_area[:,0], wide_area[:,1], s=3, color="gray", alpha=0.2)
plt.scatter(grid_coords[:,0], grid_coords[:,1], s=3, color="red", alpha=0.5)
plt.show()
