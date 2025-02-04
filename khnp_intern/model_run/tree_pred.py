import numpy as np
from data_process.data_path import estimate_result_path
from data_process.krig_data import KrigPredData
from model.tree import TrainTestTree


if __name__ == "__main__":
    cols = ["genHour", "eff_krig", "TA", "RN", "WS", "WD", "HM", "PV", "TD", 
               "PA", "PS", "SS", "SI", "SD_TOT", "SD_HR3", "CA_TOT", "CA_MID", 
               "VS", "TS", "SW038", "WV063", "WV069", "WV073", "IR087", "IR096", 
               "IR105", "IR112", "IR123", "IR133"]

    
    krig_data = KrigPredData(cols)
    model_name = "GBR"

    train_input, train_label = krig_data.read_train_data()
    model_train_test = TrainTestTree(model_name)
    model_train_test.train(train_input, train_label)

    for estimate_file in krig_data.estimate_files:
        original_dt, estimate_input = krig_data.read_estimate_data(estimate_file)
        estimate_val = model_train_test.predict(estimate_input)
        result_df = original_dt[["genHour", "lat", "lon"]].copy()
        result_df["efficiency"] = estimate_val
        save_file = estimate_file.split(".")[0]
        save_path = f"{estimate_result_path}{model_name}/{save_file}.parquet"
        result_df.to_parquet(save_path)
        print(save_file)
