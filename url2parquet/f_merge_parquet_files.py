import os
import pandas as pd
from glob import glob

# function to process and save parquet
def merge_parquet_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # search all parquet files
    parquet_files = glob(os.path.join(input_directory, "*.parquet"))

    # extract date, time, and size
    grouped_files = {}
    for file_path in parquet_files:
        file_name = os.path.basename(file_path)
        parts = file_name.split("_")

    
        date = parts[3]
        time = parts[4]
        size = parts[5].replace(".parquet", "")

        # group key: date_size
        group_key = f"{date}_{size}"

        if group_key not in grouped_files:
            grouped_files[group_key] = []
        grouped_files[group_key].append((time, file_path))

    # merge files
    for group_key, files in grouped_files.items():
        files.sort()  # sort by time

        merged_data = []
        for time, file_path in files:
            df = pd.read_parquet(file_path)  # read parquet file

            # add time columns
            datetime_str = group_key.split("_")[0] 
            df.insert(0, "Datetime", datetime_str)
            df.insert(1, "Hour", time)

            merged_data.append(df)

        # sace merged data (one day parquet file)
        merged_df = pd.concat(merged_data, ignore_index=True)
        output_file = os.path.join(output_directory, f"merged_{group_key}.parquet")
        merged_df.to_parquet(output_file, index=False)  # as parquet file
    print("모든 Parquet 파일이 병합되었습니다!")