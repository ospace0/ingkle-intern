import numpy as np
import pandas as pd
from data_process.data_path import generator_path

def read_validation_set():
    set_df = pd.read_excel(f"{generator_path}validation_set.xlsx", index_col=0)
    all_set_dict = {}
    for set_id in set_df.index:
        all_set_dict[set_id] = list(set_df.loc[set_id])
    return all_set_dict


