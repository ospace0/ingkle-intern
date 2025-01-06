import xarray as xr
import matplotlib.pyplot as plt

# NetCDF 파일 열기
file_path = r"F:/INKLE/satelite/output_file.nc"  # 파일 경로에 맞게 수정
dataset = xr.open_dataset(file_path)

# 이미지 데이터 추출
if "image_pixel_values" in dataset.data_vars:
    image_data = dataset["image_pixel_values"].values  # 이미지 픽셀 값
else:
    raise KeyError("Variable 'image_pixel_values' not found in the dataset.")

# 데이터 시각화 (컬러)
plt.figure(figsize=(10, 10))
plt.imshow(image_data, cmap="jet")  # 'jet' 컬러맵 적용
plt.title("Satellite Image")
plt.colorbar(label="Pixel Value")
plt.show()

# 컬러 JPG 파일로 저장
output_image_path = r"F:/INKLE/satelite/satellite_image_color.jpg"
plt.imsave(output_image_path, image_data, cmap="jet")
print(f"Color image saved to {output_image_path}")

# Dataset 닫기
dataset.close()
