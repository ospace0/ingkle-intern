from data_process.kriging import TrainTestKriging

if __name__ == "__main__":
    kriging = TrainTestKriging()
    kriging.daily_krig()
