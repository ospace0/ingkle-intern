import os
import time
import shutil

from datetime import timedelta
from tqdm import tqdm
from changable_things import data_types, start_date, end_date, parquet_directory, nc_file_path, merged_directory, size_ranges, download_period, total_steps
from f_merge_parquet_files import merge_parquet_files
from f_process_and_save_parquet import process_and_save_parquet
from url2parquet.f_nc_read import process_data_types

#main function(run this function not f_process_and_save_parquet.py, f_merge_parquet_files.py or f_nc_read.py)
def main():
    start_time = time.time()  # start time check
   
    # Mode selection
    mode = input("Select mode: 1(get from url) or 2(read downloaded nc)\n")
    if mode == '1':
        get_nc = "url"
    elif mode == '2':
        get_nc = nc_file_path
    else:
        print("Invalid mode selected.")
        return
    
    # tqdm progress bar for all dates
    date_progress = tqdm(total=(end_date - start_date).days, desc="전체 진행도 (Dates)")

    searching_date = start_date
    while searching_date < end_date:  # (1) all dates loop
        hour_progress = tqdm(total=24, desc=f"{searching_date.strftime('%Y-%m-%d')} 진행도 (Hours)", leave=False)

        # make one day into 24 hours
        searching_hours = [searching_date + timedelta(hours=hour) for hour in range(24)]
        for searching_hour in searching_hours:  # (2) one day, all 24 hours loop

            time_progress = tqdm(total=total_steps, desc=f"{searching_hour.strftime('%H:%M')} 진행도", leave=False)
            searching_time = searching_hour
            hourly_data_list = []
            while searching_time < searching_hour + timedelta(hours=1):  # (3) 1 hour loop
                
                file_data_per_type = process_data_types(get_nc, data_types, searching_time) # (4) data types loop, for 16 data types(wavelengths)

                hourly_data_list.append(file_data_per_type)
                searching_time += timedelta(minutes=download_period)
                time.sleep(1)

                time_progress.update(1)
            time_progress.close()

            hour_progress.update(1)

            # save parquet files
            timestamp = f"{searching_hour.year}{searching_hour.month:02}{searching_hour.day:02}_{searching_hour.hour}"
            parquet_base_path = f"{parquet_directory}/satellite_data_1havg_{timestamp}"
            process_and_save_parquet(hourly_data_list, data_types, parquet_base_path, size_ranges)

        hour_progress.close()

        date_progress.update(1)
        searching_date += timedelta(days=1)

        merge_parquet_files(parquet_directory, merged_directory)

        # hourly paquet files delete
        for filename in os.listdir(parquet_directory):
            file_path = os.path.join(parquet_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('파일 삭제 중 오류 발생: %s' % e)
        print("Parquet 디렉토리 내 파일들이 성공적으로 삭제되었습니다.")


    date_progress.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"전체 과정 완료! 총 걸린 시간: {timedelta(seconds=elapsed_time)}")

if __name__ == '__main__':
    main()