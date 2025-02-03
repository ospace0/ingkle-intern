import numpy as np
from sklearn.metrics import r2_score
from data_process.krig_data import KrigTrainTestData
from model.tree import TrainTestTree

if __name__ == "__main__":
    all_cols = ["genHour", "eff_krig", "TA", "RN", "WS", "WD", "HM", "PV", "TD", 
               "PA", "PS", "SS", "SI", "SD_TOT", "SD_HR3", "CA_TOT", "CA_MID", 
               "VS", "TS", "SW038", "WV063", "WV069", "WV073", "IR087", "IR096", 
               "IR105", "IR112", "IR123", "IR133"]

    weather_cols = ["genHour", "TA", "RN", "WS", "WD", "HM", "PV", "TD",
                   "PA", "PS", "SS", "SI", "SD_TOT", "SD_HR3", "CA_TOT", "CA_MID", 
                   "VS", "TS"]

    satellite_cols = ["genHour", "SW038", "WV063", "WV069", "WV073", "IR087", "IR096", 
                   "IR105", "IR112", "IR123", "IR133"]
    
    cols_list = [all_cols, weather_cols, satellite_cols]

    # test_cols = ["genHour", "eff_krig", "RN"]
    # cols_list = [test_cols]

    for cols in cols_list: 
        krig_data = KrigTrainTestData(cols)
        model_name = "GBR"
    
        set_acc, set_day_acc = {}, {}
        for set_name in krig_data.all_sets:
            print(set_name)
            train_input, train_label, valid_input, valid_label, valid_df = krig_data.read_set_data(set_name)
            model_train_test = TrainTestTree(model_name)
            model_train_test.train(train_input, train_label)
            valid_pred = model_train_test.predict(valid_input)
            day_label = valid_label[(valid_df["genHour"]>=10) & (valid_df["genHour"]<=16)]
            day_pred = valid_pred[(valid_df["genHour"]>=10) & (valid_df["genHour"]<=16)]
            set_acc[set_name] = r2_score(valid_label, valid_pred)
            set_day_acc[set_name] = r2_score(day_label, day_pred)
        print(set_acc)
        print(set_day_acc)
        print(round(np.mean(list(set_acc.values())), 3))
        print(round(np.mean(list(set_day_acc.values())), 3))
