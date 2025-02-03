import os
import numpy as np
import pandas as pd
from data_process.data_path import estimation_path, kriging_path


class DataStatistics:
    def _compute_statistics(self, train_data: pd.DataFrame):
        mean, std = train_data.mean(axis=0), train_data.std(axis=0)
        var_cols = train_data.columns[std > 0]
        self.mean = mean[var_cols]
        self.std = std[var_cols]
        statsitic_df = pd.concat([self.mean, self.std], axis=1)
        statsitic_df.columns = ["mean", "std"]
        if not os.path.exists("trained_models/"):
            os.makedirs("trained_models/")
        statsitic_df.to_csv("trained_models/statistics.csv")
        return var_cols
    
    def _load_statistics(self):
        statsitic_df = pd.read_csv("trained_models/statistics.csv", index_col=0)
        self.mean = statsitic_df["mean"]
        self.std = statsitic_df["std"]
        return statsitic_df.index

    def _normalize_dataset(self, dataset: pd.DataFrame, variable_cols: list):
        normalized_dt = (dataset[variable_cols] - self.mean) / self.std
        normalized_arr = np.array(normalized_dt, dtype=np.float64)
        return normalized_arr

class KrigTrainTestData(DataStatistics):
    def __init__(self, input_cols: list):
        self.input_cols = input_cols
        self.label_col = ["efficiency"]
        self.all_sets = sorted(os.listdir(kriging_path))

    def read_set_data(self, set_id: str):
        set_dir = f"{kriging_path}{set_id}/"
        date_files = sorted(os.listdir(set_dir))

        all_train_input, all_train_label = [], []
        all_valid_input, all_valid_label = [], []
        for date_f in date_files:
            date_npz = np.load(f"{set_dir}{date_f}", allow_pickle=True)
            train_df = pd.DataFrame(date_npz["train"], columns=date_npz["columns"])
            valid_df = pd.DataFrame(date_npz["valid"], columns=date_npz["columns"])
            all_train_input.append(train_df[self.input_cols])
            all_train_label.append(train_df[self.label_col])
            all_valid_input.append(valid_df[self.input_cols])
            all_valid_label.append(valid_df[self.label_col])
        all_train_input = pd.concat(all_train_input)
        all_train_label = np.array(pd.concat(all_train_label), dtype=np.float64).flatten()
        all_valid_df = pd.concat(all_valid_input)
        all_valid_label = np.array(pd.concat(all_valid_label), dtype=np.float64).flatten()
        variable_cols = self._compute_statistics(all_train_input)
        all_train_input = self._normalize_dataset(all_train_input, variable_cols)
        all_valid_input = self._normalize_dataset(all_valid_df, variable_cols)
        return all_train_input, all_train_label, all_valid_input, all_valid_label, all_valid_df
    
class KrigPredData(DataStatistics):
    def __init__(self):
        # self.input_cols = _input_cols
        self.label_col = ["efficiency"]
        self.estimate_dir = f"{estimation_path}kriging_input/estimation/"
        self.estimate_files = sorted(os.listdir(self.estimate_dir))

    def read_train_data(self):
        train_dir = f"{estimation_path}kriging_input/train/"
        train_files = sorted(os.listdir(train_dir))

        all_train_input, all_train_label = [], []
        for date_f in train_files:
            train_df = pd.read_parquet(f"{train_dir}{date_f}")
            all_train_input.append(train_df[self.input_cols])
            all_train_label.append(train_df[self.label_col])
        all_train_input = pd.concat(all_train_input)
        all_train_label = np.array(pd.concat(all_train_label), dtype=np.float64).flatten()
        variable_cols = self._compute_statistics(all_train_input)
        all_train_input = self._normalize_dataset(all_train_input, variable_cols)
        return all_train_input, all_train_label

    def read_estimate_data(self, file_name: str):
        estimate_df = pd.read_parquet(f"{self.estimate_dir}{file_name}")
        variable_cols = self._load_statistics()
        estimate_input = self._normalize_dataset(estimate_df, variable_cols)
        return estimate_df, estimate_input

class KrigRealtimeData(DataStatistics):
    def __init__(self):
        # self.input_cols = _input_cols
        self.label_col = ["efficiency"]

    def normalize_data(self, estimate_df: pd.DataFrame):
        variable_cols = self._load_statistics()
        estimate_input = self._normalize_dataset(estimate_df, variable_cols)
        return estimate_input
