from data_process.kriging import EstimateKriging, WholeTrainKriging

if __name__ == "__main__":
    estimate_kriging = EstimateKriging()
    estimate_kriging.daily_krig()
    whole_kriging = WholeTrainKriging()
    whole_kriging.daily_krig()
