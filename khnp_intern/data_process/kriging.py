import os
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from data_process.data_path import generator_path, kriging_path, estimation_path
from data_process.split import read_validation_set
from data_process.original_data import OriginalData

class TrainTestKriging:
    def __init__(self):
        self.validation_set = read_validation_set()
        self.data_files = self._list_dates()
        self.original_data = OriginalData()

    def _list_dates(self):
        generator_files = sorted(os.listdir(f"{generator_path}daily/"))
        return generator_files
    
    def _date_gen_krig(self, gen_dt: pd.DataFrame, set_valid_gens: list, set_train_gens: list):
        valid_gen_dt = gen_dt[np.isin(gen_dt["kpxGenid"], set_valid_gens)]
        train_gen_dt = gen_dt[np.isin(gen_dt["kpxGenid"], set_train_gens)]
        valid_krig, train_krig = [], []
        for h in range(1, 24):
            hour_train_gen = train_gen_dt[train_gen_dt["genHour"]==h].copy()
            hour_valid_gen = valid_gen_dt[valid_gen_dt["genHour"]==h].copy()
            if hour_train_gen["efficiency"].std() == 0:
                hour_valid_gen["eff_krig"] = hour_train_gen["efficiency"].mean()
                hour_train_gen["eff_krig"] = hour_train_gen["efficiency"].mean()
            else:
                try:
                    train_whole_uk = OrdinaryKriging(hour_train_gen["lon"], hour_train_gen["lat"], hour_train_gen["efficiency"])
                    hour_valid_krig, _ = train_whole_uk.execute("points", hour_valid_gen["lon"], hour_valid_gen["lat"])
                    if np.isnan(hour_valid_krig).any():
                        raise
                    hour_valid_krig = np.minimum(np.maximum(hour_valid_krig, min(hour_train_gen["efficiency"])), max(hour_train_gen["efficiency"]))
                    hour_valid_gen["eff_krig"] = np.array(hour_valid_krig)
                    hour_train_krig = []
                    for tr_dt_id in hour_train_gen.index:
                        id_krig_df = hour_train_gen.loc[[tr_dt_id]].copy()
                        krig_train_data = hour_train_gen.drop([tr_dt_id], axis=0)
                        if krig_train_data["efficiency"].std() == 0:
                            id_krig_df["eff_krig"] = krig_train_data["efficiency"].mean()
                        else:
                            try:
                                train_set_uk = OrdinaryKriging(krig_train_data["lon"], krig_train_data["lat"], krig_train_data["efficiency"])
                                hour_id_train_krig, _ = train_set_uk.execute("points", hour_train_gen.loc[tr_dt_id, "lon"], hour_train_gen.loc[tr_dt_id, "lat"])
                                if np.isnan(hour_id_train_krig).any():
                                    raise
                                hour_id_train_krig = np.minimum(np.maximum(hour_id_train_krig, min(krig_train_data["efficiency"])), max(krig_train_data["efficiency"]))
                                id_krig_df["eff_krig"] = np.array(hour_id_train_krig)
                            except:
                                val, count = np.unique(krig_train_data["efficiency"], return_counts=True)
                                id_krig_df["eff_krig"] = val[np.argmax(count)]
                        hour_train_krig.append(id_krig_df)
                    hour_train_gen = pd.concat(hour_train_krig, ignore_index=True)
                except:
                    val, count = np.unique(hour_train_gen["efficiency"], return_counts=True)
                    unified_val = val[np.argmax(count)]
                    hour_valid_gen["eff_krig"] = unified_val
                    hour_train_gen["eff_krig"] = unified_val
            if hour_valid_gen.isnull().values.any():
                raise Exception("Nan value in df.")
            if hour_train_gen.isnull().values.any():
                raise Exception("Nan value in df.")
            hour_valid_gen = hour_valid_gen.set_index(["kpxGenid", "genHour", "lat", "lon"])
            hour_train_gen = hour_train_gen.set_index(["kpxGenid", "genHour", "lat", "lon"])
            valid_krig.append(hour_valid_gen.copy())
            train_krig.append(hour_train_gen.copy())

        valid_krig_df = pd.concat(valid_krig)
        train_krig_df = pd.concat(train_krig)
        if train_krig_df.isnull().values.any():
            raise("Nan value in df.")
        if valid_krig_df.isnull().values.any():
            raise("Nan value in df.")
        return train_krig_df, valid_krig_df

    def _date_weather_krig(self, weather_dt: pd.DataFrame, gen_dt: pd.DataFrame, set_valid_gens: list, set_train_gens: list,
                           krig_columns: list):
        valid_gen_dt = gen_dt[np.isin(gen_dt["kpxGenid"], set_valid_gens)]
        train_gen_dt = gen_dt[np.isin(gen_dt["kpxGenid"], set_train_gens)]
        valid_krig, train_krig = [], []
        for h in range(1, 24):
            hour_weather = weather_dt[weather_dt["time"]==h]
            hour_train_gen = train_gen_dt[train_gen_dt["genHour"]==h]
            hour_valid_gen = valid_gen_dt[valid_gen_dt["genHour"]==h]
            hour_train_weather = pd.DataFrame(index=hour_train_gen.index, columns=krig_columns)
            hour_valid_weather = pd.DataFrame(index=hour_valid_gen.index, columns=krig_columns)
            hour_valid_krig = pd.concat([hour_valid_gen[["kpxGenid", "genHour", "lat", "lon"]], hour_valid_weather], axis=1)
            hour_train_krig = pd.concat([hour_train_gen[["kpxGenid", "genHour", "lat", "lon"]], hour_train_weather], axis=1)
            for w_col in krig_columns:
                hour_w_weather = hour_weather[["lon", "lat", w_col]].dropna(subset=[w_col])
                if hour_weather[w_col].std() == 0:
                    hour_valid_krig.loc[:, w_col] = hour_weather[w_col].mean()
                    hour_train_krig.loc[:, w_col] = hour_weather[w_col].mean()
                else:
                    try:
                        weather_uk = OrdinaryKriging(hour_w_weather["lon"], hour_w_weather["lat"], hour_w_weather[w_col])
                        hour_w_valid_krig, _ = weather_uk.execute("points", hour_valid_gen["lon"], hour_valid_gen["lat"])
                        if np.isnan(hour_w_valid_krig).any():
                            raise
                        hour_w_train_krig, _ = weather_uk.execute("points", hour_train_gen["lon"], hour_train_gen["lat"])
                        hour_w_valid_krig = np.minimum(np.maximum(hour_w_valid_krig, min(hour_w_weather[w_col])), max(hour_w_weather[w_col]))
                        hour_w_train_krig = np.minimum(np.maximum(hour_w_train_krig, min(hour_w_weather[w_col])), max(hour_w_weather[w_col]))
                        hour_valid_krig.loc[:, w_col] = np.array(hour_w_valid_krig)
                        hour_train_krig.loc[:, w_col] = np.array(hour_w_train_krig)
                    except:
                        val, count = np.unique(hour_weather[w_col], return_counts=True)
                        unified_val = val[np.argmax(count)]
                        hour_valid_krig.loc[:, w_col] = unified_val
                        hour_train_krig.loc[:, w_col] = unified_val
                if hour_train_krig.loc[:, w_col].isnull().values.any():
                    raise Exception("Nan value in df.")
                if hour_valid_krig.loc[:, w_col].isnull().values.any():
                    raise Exception("Nan value in df.")
            hour_valid_krig = hour_valid_krig.set_index(["kpxGenid", "genHour", "lat", "lon"])
            hour_train_krig = hour_train_krig.set_index(["kpxGenid", "genHour", "lat", "lon"])
            valid_krig.append(hour_valid_krig)
            train_krig.append(hour_train_krig)

        valid_krig_df = pd.concat(valid_krig)
        train_krig_df = pd.concat(train_krig)
        if train_krig_df.isnull().values.any():
            raise Exception("Nan value in df.")
        if valid_krig_df.isnull().values.any():
            raise Exception("Nan value in df.")
        return train_krig_df, valid_krig_df

    def _wa_satellite(self, satellite_dt: pd.DataFrame, gen_dt: pd.DataFrame, krig_columns: list):
        gen_ids = gen_dt.drop_duplicates(subset=["kpxGenid"])[["kpxGenid", "lat", "lon"]]
        krig_data_cols = ["kpxGenid", "genHour", "lat", "lon"] + krig_columns
        extract_range = 0.1
        all_krig = []
        for g_id in gen_ids.index:
            kpx_id, lat, lon  = gen_ids.loc[g_id] 
            range_satellite_data = satellite_dt[
                (satellite_dt['lat'] >= lat - extract_range) & (satellite_dt['lat'] <= lat + extract_range) &
                (satellite_dt['lon'] >= lon - extract_range) & (satellite_dt['lon'] <= lon + extract_range)]
            range_satellite_data = range_satellite_data.sort_values(["lat", "lon"])
            first_hour_data = range_satellite_data[range_satellite_data["hour"]==0]
            distances = np.sqrt((first_hour_data['lat'] - lat) ** 2 + (first_hour_data['lon'] - lon) ** 2)
            weights = 1 / distances
            normalized_weights = np.array(weights / weights.sum())

            exist_hours = np.array(gen_dt[gen_dt['kpxGenid']==kpx_id]["genHour"])
            hour_krig_data = pd.DataFrame(columns=krig_data_cols, index=range(len(exist_hours)))
            hour_krig_data.loc[:, ['lat', 'lon', 'kpxGenid']] = [lat, lon, kpx_id]
            hour_krig_data.loc[:, "genHour"] = exist_hours
            hour_krig_data = hour_krig_data.set_index("genHour")
            for h in exist_hours:
                hour_satellite_data = range_satellite_data[range_satellite_data["hour"]==h]
                for w_col in krig_columns:
                    wa_val = (np.array(hour_satellite_data[w_col]) * normalized_weights).sum()
                    hour_krig_data.loc[h, w_col] = wa_val
            hour_krig_data = hour_krig_data.reset_index(drop=False)
            all_krig.append(hour_krig_data.copy())
        all_krig_df = pd.concat(all_krig, ignore_index=True)
        all_krig_df = all_krig_df.astype({"lat": "float64", "lon": "float64"})
        all_krig_df = all_krig_df.set_index(["kpxGenid", "genHour", "lat", "lon"])
        return all_krig_df[krig_columns]

    def _date_satellite_krig(self, satellite_dt: pd.DataFrame, gen_dt: pd.DataFrame, set_valid_gens: list, set_train_gens: list, krig_satellite_columns: list):
        valid_gen_dt = gen_dt[np.isin(gen_dt["kpxGenid"], set_valid_gens)]
        train_gen_dt = gen_dt[np.isin(gen_dt["kpxGenid"], set_train_gens)]

        valid_krig_df = self._wa_satellite(satellite_dt, valid_gen_dt, krig_satellite_columns)
        train_krig_df = self._wa_satellite(satellite_dt, train_gen_dt, krig_satellite_columns)
        return train_krig_df, valid_krig_df

    def _date_krig(self, date_str: str):
        gen_dt = self.original_data.convert_generator(f"{date_str}")
        weather_dt = self.original_data.convert_weather(f"{date_str}")
        satellite_dt = self.original_data.convert_satellite(f"{date_str}")
        krig_satellite_columns = [c for c in satellite_dt.columns if c not in ["hour", "datetime", "lat", "lon"]]
        krig_weather_columns = [c for c in weather_dt.columns if c not in ["time", "lat", "lon"]]
        for set_id in self.validation_set.keys():
            set_valid_gens = [str(g) for g in self.validation_set[set_id] if not pd.isnull(g)]
            set_train_gens = [i for i in np.unique(gen_dt["kpxGenid"]) if i not in set_valid_gens]
            train_gen, valid_gen = self._date_gen_krig(gen_dt, set_valid_gens, set_train_gens)
            train_weather, valid_weather = self._date_weather_krig(weather_dt, gen_dt, set_valid_gens, set_train_gens, krig_weather_columns)
            train_satellite, valid_satellite = self._date_satellite_krig(satellite_dt, gen_dt, set_valid_gens, set_train_gens, krig_satellite_columns)
            set_train_krig_df = pd.concat([train_gen, train_weather, train_satellite], axis=1).reset_index(drop=False)
            set_valid_krig_df = pd.concat([valid_gen, valid_weather, valid_satellite], axis=1).reset_index(drop=False)
            if set_train_krig_df.isnull().values.any():
                raise Exception("Nan value in df.")
            if set_valid_krig_df.isnull().values.any():
                raise Exception("Nan value in df.")
            save_dir = f"{kriging_path}{set_id}/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.savez(f"{save_dir}{date_str}.npz",
                     train=np.array(set_train_krig_df), valid=np.array(set_valid_krig_df),
                     columns=np.array(set_train_krig_df.columns))

    def daily_krig(self):
        save_dir = f"{kriging_path}set4/"
        for date_file in self.data_files:
            unite_file_name = date_file.split(".")[0]
            save_file_name = f"{unite_file_name}.npz"
            if save_file_name in os.listdir(save_dir):
                continue
            print(unite_file_name)
            self._date_krig(unite_file_name)

class WholeTrainKriging:
    def __init__(self):
        self.data_files = self._list_dates()
        self.original_data = OriginalData()

    def _list_dates(self):
        generator_files = sorted(os.listdir(f"{generator_path}daily/"))
        return generator_files

    def _date_gen_krig(self, gen_dt: pd.DataFrame):
        train_krig = []
        for h in range(1, 24):
            hour_train_gen = gen_dt[gen_dt["genHour"]==h].copy()
            if hour_train_gen["efficiency"].std() == 0:
                hour_train_gen["eff_krig"] = hour_train_gen["efficiency"].mean()
            else:
                hour_train_krig = []
                for tr_dt_id in hour_train_gen.index:
                    id_krig_df = hour_train_gen.loc[[tr_dt_id]].copy()
                    krig_train_data = hour_train_gen.drop([tr_dt_id], axis=0)
                    if krig_train_data["efficiency"].std() == 0:
                        id_krig_df["eff_krig"] = krig_train_data["efficiency"].mean()
                    else:
                        try:
                            train_set_uk = OrdinaryKriging(krig_train_data["lon"], krig_train_data["lat"], krig_train_data["efficiency"])
                            hour_id_train_krig, _ = train_set_uk.execute("points", hour_train_gen.loc[tr_dt_id, "lon"], hour_train_gen.loc[tr_dt_id, "lat"])
                            if np.isnan(hour_id_train_krig).any():
                                raise
                            hour_id_train_krig = np.minimum(np.maximum(hour_id_train_krig, min(krig_train_data["efficiency"])), max(krig_train_data["efficiency"]))
                            id_krig_df["eff_krig"] = np.array(hour_id_train_krig)
                        except:
                            val, count = np.unique(krig_train_data["efficiency"], return_counts=True)
                            id_krig_df["eff_krig"] = val[np.argmax(count)]
                    hour_train_krig.append(id_krig_df)
                hour_train_gen = pd.concat(hour_train_krig, ignore_index=True)
            if hour_train_gen.isnull().values.any():
                raise Exception("Nan value in df.")
            train_krig.append(hour_train_gen.copy())
        train_krig_df = pd.concat(train_krig, ignore_index=True)
        train_krig_df = train_krig_df.set_index(["genHour", "kpxGenid"])
        return train_krig_df

    def _date_weather_krig(self, weather_dt: pd.DataFrame, gen_dt: pd.DataFrame, krig_columns: list):
        train_krig = []
        for h in range(1, 24):
            hour_weather = weather_dt[weather_dt["time"]==h]
            hour_train_gen = gen_dt[gen_dt["genHour"]==h]
            hour_train_weather = pd.DataFrame(index=hour_train_gen.index, columns=krig_columns)
            hour_train_krig = pd.concat([hour_train_gen[["kpxGenid", "genHour", "lat", "lon"]], hour_train_weather], axis=1)
            for w_col in krig_columns:
                hour_w_weather = hour_weather[["lon", "lat", w_col]].dropna(subset=[w_col])
                if hour_weather[w_col].std() == 0:
                    hour_train_krig.loc[:, w_col] = hour_weather[w_col].mean()
                else:
                    try:
                        weather_uk = OrdinaryKriging(hour_w_weather["lon"], hour_w_weather["lat"], hour_w_weather[w_col])
                        hour_w_train_krig, _ = weather_uk.execute("points", hour_train_gen["lon"], hour_train_gen["lat"])
                        hour_w_train_krig = np.minimum(np.maximum(hour_w_train_krig, min(hour_w_weather[w_col])), max(hour_w_weather[w_col]))
                        hour_train_krig.loc[:, w_col] = np.array(hour_w_train_krig)
                        if np.isnan(hour_w_train_krig).any():
                            raise
                    except:
                        val, count = np.unique(hour_weather[w_col], return_counts=True)
                        unified_val = val[np.argmax(count)]
                        hour_train_krig.loc[:, w_col] = unified_val
                if hour_train_krig.loc[:, w_col].isnull().values.any():
                    raise Exception("Nan value in df.")
            train_krig.append(hour_train_krig)
        train_krig_df = pd.concat(train_krig, ignore_index=True)
        train_krig_df = train_krig_df.set_index(["genHour", "kpxGenid"])
        return train_krig_df[krig_columns]

    def _date_satellite_wa(self, satellite_dt: pd.DataFrame, gen_dt: pd.DataFrame, krig_columns: list):
        gen_ids = gen_dt.drop_duplicates(subset=["kpxGenid"])[["kpxGenid", "lat", "lon"]]
        krig_data_cols = ["kpxGenid", "genHour", "lat", "lon"] + krig_columns
        extract_range = 0.1
        all_krig = []
        for g_id in gen_ids.index:
            kpx_id, lat, lon  = gen_ids.loc[g_id] 
            range_satellite_data = satellite_dt[
                (satellite_dt['lat'] >= lat - extract_range) & (satellite_dt['lat'] <= lat + extract_range) &
                (satellite_dt['lon'] >= lon - extract_range) & (satellite_dt['lon'] <= lon + extract_range)]
            range_satellite_data = range_satellite_data.sort_values(["lat", "lon"])
            first_hour_data = range_satellite_data[range_satellite_data["hour"]==0]
            distances = np.sqrt((first_hour_data['lat'] - lat) ** 2 + (first_hour_data['lon'] - lon) ** 2)
            weights = 1 / distances
            normalized_weights = np.array(weights / weights.sum())

            exist_hours = np.array(gen_dt[gen_dt['kpxGenid']==kpx_id]["genHour"])
            hour_krig_data = pd.DataFrame(columns=krig_data_cols, index=range(len(exist_hours)))
            hour_krig_data.loc[:, ['lat', 'lon', 'kpxGenid']] = [lat, lon, kpx_id]
            hour_krig_data.loc[:, "genHour"] = exist_hours
            hour_krig_data = hour_krig_data.set_index("genHour")
            for h in exist_hours:
                hour_satellite_data = range_satellite_data[range_satellite_data["hour"]==h]
                for w_col in krig_columns:
                    wa_val = (np.array(hour_satellite_data[w_col]) * normalized_weights).sum()
                    hour_krig_data.loc[h, w_col] = wa_val
            hour_krig_data = hour_krig_data.reset_index(drop=False)
            all_krig.append(hour_krig_data.copy())
        all_krig_df = pd.concat(all_krig, ignore_index=True)
        all_krig_df = all_krig_df.set_index(["genHour", "kpxGenid"])
        return all_krig_df[krig_columns]
    
    def _date_krig(self, date_file: str):
        gen_dt = self.original_data.convert_generator(date_file)
        weather_dt = self.original_data.convert_weather(date_file)
        satellite_dt = self.original_data.convert_satellite(date_file)
        krig_weather_columns = [c for c in weather_dt.columns if c not in ["time", "lat", "lon"]]
        krig_satellite_columns = [c for c in satellite_dt.columns if c not in ["hour", "datetime", "lat", "lon"]]   
        gen_krig = self._date_gen_krig(gen_dt)
        weather_krig = self._date_weather_krig(weather_dt, gen_dt, krig_weather_columns)
        satellite_wa = self._date_satellite_wa(satellite_dt, gen_dt, krig_satellite_columns)
        combined_data = pd.concat([gen_krig, weather_krig, satellite_wa], axis=1).reset_index(drop=False)
        if combined_data.isnull().values.any():
            raise Exception("Nan value in df.")
        save_dir = f"{estimation_path}kriging_input/train/" 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file_name = date_file.split(".")[0]
        combined_data.to_parquet(f"{save_dir}{save_file_name}.parquet")

    def daily_krig(self):
        save_dir = f"{estimation_path}kriging_input/train/"
        for date_file in self.data_files:
            date_file_name = date_file.split(".")[0]
            save_file_name = f"{date_file_name}.parquet"
            # if save_file_name in os.listdir(save_dir):
            #     continue
            print(date_file_name)
            self._date_krig(date_file_name)

class EstimateKriging:
    def __init__(self):
        self.pred_coords = pd.read_excel(f"{estimation_path}estimate_locations.xlsx")
        self.data_files = self._list_dates()
        self.original_data = OriginalData()

    def _list_dates(self):
        generator_files = sorted(os.listdir(f"{generator_path}daily/"))
        return generator_files
    
    def _date_gen_krig(self, gen_dt: pd.DataFrame):
        gen_dt_cols = ["genHour", "lat", "lon", "eff_krig"]
        all_krig = []
        for h in range(1, 24):
            hour_krig_data = pd.DataFrame(columns=gen_dt_cols, index=range(len(self.pred_coords)))
            hour_krig_data.loc[:, ["lat", "lon"]] = self.pred_coords
            hour_krig_data = hour_krig_data.astype({"lat": 'float64', "lon": "float64"})
            hour_krig_data.loc[:, "genHour"] = h
            hour_gen_dt = gen_dt[gen_dt["genHour"]==h].copy()
            if hour_gen_dt["efficiency"].std() == 0:
                hour_krig_data["eff_krig"] = hour_gen_dt["efficiency"].mean()
            else:
                try:
                    whole_uk = OrdinaryKriging(hour_gen_dt["lon"], hour_gen_dt["lat"], hour_gen_dt["efficiency"])
                    hour_krig, _ = whole_uk.execute("points", hour_krig_data["lon"], hour_krig_data["lat"])
                    if np.isnan(hour_krig).any():
                        raise
                    hour_krig = np.minimum(np.maximum(hour_krig, min(hour_gen_dt["efficiency"])), max(hour_gen_dt["efficiency"]))
                    hour_krig_data["eff_krig"] = np.array(hour_krig)
                except:
                    val, count = np.unique(hour_gen_dt["efficiency"], return_counts=True)
                    unified_val = val[np.argmax(count)]
                    hour_krig_data["eff_krig"] = unified_val
            if hour_krig_data.isnull().values.any():
                raise Exception("Nan value in df.")
            all_krig.append(hour_krig_data.copy())
        all_krig_df = pd.concat(all_krig, ignore_index=True)
        all_krig_df = all_krig_df.set_index(["genHour", "lat", "lon"])
        return all_krig_df

    def _date_weather_krig(self, weather_dt: pd.DataFrame, krig_columns: list):
        krig_data_cols = ["genHour", "lat", "lon"] + krig_columns
        all_krig = []
        for h in range(1, 24):
            hour_krig_data = pd.DataFrame(columns=krig_data_cols, index=range(len(self.pred_coords)))
            hour_krig_data.loc[:, ["lat", "lon"]] = self.pred_coords
            hour_krig_data = hour_krig_data.astype({"lat": 'float64', "lon": "float64"})
            hour_krig_data.loc[:, "genHour"] = h
            hour_weather = weather_dt[weather_dt["time"]==h]
            for w_col in krig_columns:
                hour_w_weather = hour_weather[["lon", "lat", w_col]].dropna(subset=[w_col])
                if hour_w_weather[w_col].std() == 0:
                    hour_krig_data.loc[:, w_col] = hour_w_weather[w_col].mean()
                else:
                    try:
                        weather_uk = OrdinaryKriging(hour_w_weather["lon"], hour_w_weather["lat"], hour_w_weather[w_col])
                        hour_w_krig, _ = weather_uk.execute("points", hour_krig_data["lon"], hour_krig_data["lat"])
                        if np.isnan(hour_w_krig).any():
                            raise
                        hour_w_krig = np.minimum(np.maximum(hour_w_krig, min(hour_w_weather[w_col])), max(hour_w_weather[w_col]))
                        hour_krig_data.loc[:, w_col] = np.array(hour_w_krig)
                    except:
                        val, count = np.unique(hour_w_weather[w_col], return_counts=True)
                        unified_val = val[np.argmax(count)]
                        hour_krig_data.loc[:, w_col] = unified_val
                if hour_krig_data.loc[:, w_col].isnull().values.any():
                    raise Exception("Nan value in df.")
            all_krig.append(hour_krig_data)
        all_krig_df = pd.concat(all_krig, ignore_index=True)
        all_krig_df = all_krig_df.set_index(["genHour", "lat", "lon"])
        return all_krig_df[krig_columns]
    
    def _date_satellite_weight(self, satellite_dt: pd.DataFrame, krig_columns: list):
        krig_data_cols = ['genHour', 'lat', 'lon'] + krig_columns
        extract_range = 0.1
        all_krig = []
        for loc_id in self.pred_coords.index:
            lon, lat = self.pred_coords.loc[loc_id]
            range_satellite_data = satellite_dt[
                (satellite_dt['lat'] >= lat - extract_range) & (satellite_dt['lat'] <= lat + extract_range) &
                (satellite_dt['lon'] >= lon - extract_range) & (satellite_dt['lon'] <= lon + extract_range)]
            range_satellite_data = range_satellite_data.sort_values(["lat", "lon"])
            first_hour_data = range_satellite_data[range_satellite_data["hour"]==0]
            distances = np.sqrt((first_hour_data['lat'] - lat) ** 2 + (first_hour_data['lon'] - lon) ** 2)
            weights = 1 / distances
            normalized_weights = np.array(weights / weights.sum())

            hour_krig_data = pd.DataFrame(columns=krig_data_cols, index=range(1, 24))
            hour_krig_data.loc[:, ['lat', 'lon']] = [lat, lon]
            hour_krig_data.loc[:, "genHour"] = range(1, 24)
            hour_krig_data = hour_krig_data.set_index("genHour")
            for h in range(1, 24):
                hour_satellite_data = range_satellite_data[range_satellite_data["hour"]==h]
                for w_col in krig_columns:
                    wa_val = (np.array(hour_satellite_data[w_col]) * normalized_weights).sum()
                    hour_krig_data.loc[h, w_col] = wa_val
            hour_krig_data = hour_krig_data.reset_index(drop=False)
            all_krig.append(hour_krig_data.copy())
        all_krig_df = pd.concat(all_krig, ignore_index=True)
        all_krig_df = all_krig_df.set_index(["genHour", "lat", "lon"])
        return all_krig_df[krig_columns]

    def _date_krig(self, date_file: str):
        gen_dt = self.original_data.convert_generator(date_file)
        weather_dt = self.original_data.convert_weather(date_file)
        satellite_dt = self.original_data.convert_satellite(date_file)
        krig_weather_columns = [c for c in weather_dt.columns if c not in ["time", "lat", "lon"]]
        krig_satellite_columns = [c for c in satellite_dt.columns if c not in ["hour", "datetime", "lat", "lon"]]
        gen_krig = self._date_gen_krig(gen_dt)
        weather_krig = self._date_weather_krig(weather_dt, krig_weather_columns)
        satellite_wa = self._date_satellite_weight(satellite_dt, krig_satellite_columns)
        combined_data = pd.concat([gen_krig, weather_krig, satellite_wa], axis=1).reset_index(drop=False)
        if combined_data.isnull().values.any():
            raise Exception("Nan value in df.")
        save_dir = f"{estimation_path}kriging_input/estimation/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file_name = date_file.split(".")[0]
        combined_data.to_parquet(f"{save_dir}{save_file_name}.parquet")

    def daily_krig(self):
        save_dir = f"{estimation_path}kriging_input/estimation/"
        for date_file in self.data_files:
            date_file_name = date_file.split(".")[0]
            save_file_name = f"{date_file_name}.parquet"
            # if save_file_name in os.listdir(save_dir):
            #     continue
            print(date_file_name)
            self._date_krig(date_file_name)

class RealtimeKriging:
    def __init__(self, datetime_hour: int):
        self.datetime_hour = datetime_hour
        self.pred_coords = pd.read_excel(f"{estimation_path}estimate_locations.xlsx")
        self.original_data = OriginalData()

    def _gen_krig(self, gen_dt: pd.DataFrame):
        gen_dt_cols = ["genHour", "lat", "lon", "eff_krig"]
        krig_data = pd.DataFrame(columns=gen_dt_cols, index=range(len(self.pred_coords)))
        krig_data.loc[:, ["lat", "lon"]] = self.pred_coords
        krig_data = krig_data.astype({"lat": 'float64', "lon": "float64"})
        krig_data.loc[:, "genHour"] = self.datetime_hour
        if gen_dt["efficiency"].std() == 0:
            krig_data["eff_krig"] = gen_dt["efficiency"].mean()
        else:
            try:
                whole_uk = OrdinaryKriging(gen_dt["lon"], gen_dt["lat"], gen_dt["efficiency"])
                krig, _ = whole_uk.execute("points", krig_data["lon"], krig_data["lat"])
                if np.isnan(krig).any():
                    raise
                krig = np.minimum(np.maximum(krig, min(gen_dt["efficiency"])), max(gen_dt["efficiency"]))
                krig_data["eff_krig"] = np.array(krig)
            except:
                val, count = np.unique(gen_dt["efficiency"], return_counts=True)
                unified_val = val[np.argmax(count)]
                krig_data["eff_krig"] = unified_val
        return krig_data

    def _weather_krig(self, weather_dt: pd.DataFrame, krig_columns: list):
        krig_data_cols = ["genHour", "lat", "lon"] + krig_columns
        krig_data = pd.DataFrame(columns=krig_data_cols, index=range(len(self.pred_coords)))
        krig_data.loc[:, ["lat", "lon"]] = self.pred_coords
        krig_data = krig_data.astype({"lat": 'float64', "lon": "float64"})
        krig_data.loc[:, "genHour"] = self.datetime_hour
        for w_col in krig_columns:
            w_weather = weather_dt[["lon", "lat", w_col]].dropna(subset=[w_col])
            if len(w_weather) < 1:
                krig_data.loc[:, w_col] = 0
            elif w_weather[w_col].std() == 0:
                krig_data.loc[:, w_col] = w_weather.mean()
            else:
                try:
                    weather_uk = OrdinaryKriging(w_weather["lon"], w_weather["lat"], w_weather[w_col])
                    w_krig, _ = weather_uk.execute("points", krig_data["lon"], krig_data["lat"])
                    if np.isnan(w_krig).any():
                        raise
                    w_krig = np.minimum(np.maximum(w_krig, min(w_weather[w_col])), max(w_weather[w_col]))
                    krig_data.loc[:, w_col] = np.array(w_krig)
                except:
                    val, count = np.unique(w_weather[w_col], return_counts=True)
                    unified_val = val[np.argmax(count)]
                    krig_data.loc[:, w_col] = unified_val
            if krig_data.loc[:, w_col].isnull().values.any():
                raise Exception("Nan value in df.")
        return krig_data

    def hour_krig(self, gen_dt: pd.DataFrame, weather_dt: pd.DataFrame):
        krig_weather_columns = [c for c in weather_dt.columns if c not in ["time", "lat", "lon"]]
        gen_krig = self._gen_krig(gen_dt)
        weather_krig = self._weather_krig(weather_dt, krig_weather_columns)
        combined_data = pd.concat([gen_krig, weather_krig[krig_weather_columns]], axis=1).reset_index(drop=True)
        return combined_data

