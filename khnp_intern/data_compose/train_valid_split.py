import pandas as pd
import numpy as np

if __name__=='__main__':
    valid_gen_num = 9
    np.random.seed(1000)
    location_data = pd.read_excel("sample data/generator/data_locations.xlsx")
    gen_id_list = list(location_data["kpxGenid"].astype(str))
    valid_gen_ids = []
    while len(gen_id_list) > 0:
        select_ids = np.random.choice(gen_id_list, valid_gen_num, False)
        valid_gen_ids.append(list(select_ids))
        gen_id_list = [i for i in gen_id_list if i not in select_ids]
    valid_gen_ids = pd.DataFrame(valid_gen_ids, 
                                 index=[f"set{i}" for i in range(len(valid_gen_ids))])
    valid_gen_ids.to_excel("sample data/generator/validation_set.xlsx")

# station 나눠야함. traintest하려면 그 작업한거거