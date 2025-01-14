import xarray as xr
import pandas as pd

# NetCDF 파일 열기
file_path = r"F:/INKLE/satelite/output_file.nc"  # 파일 경로 수정
dataset = xr.open_dataset(file_path)

# 파일 구조 확인 (변수 이름 파악)
print("Variables in the dataset:")
print(dataset)

# 필요한 데이터 선택 (예: 'image_pixel_values')
if "image_pixel_values" in dataset.data_vars:
    variable_data = dataset["image_pixel_values"]
else:
    raise KeyError("Variable 'image_pixel_values' not found in the dataset.")

# 데이터 평면화 (CSV로 저장하기 위한 2D 구조로 변환)
df = variable_data.to_dataframe().reset_index()

# CSV 파일로 저장
output_csv_path = r"F:/INKLE/satelite/output_data.csv"
df.to_csv(output_csv_path, index=False)
print(f"Data saved to {output_csv_path}")

# Dataset 닫기
dataset.close()
b=3