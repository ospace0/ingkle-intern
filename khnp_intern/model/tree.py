import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

class TrainTestTree:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _define_rf(self):
        params = dict({
            "n_estimators": 40,
            "max_leaf_nodes": 4,
            "max_depth": None,
            "random_state": 2,
            "min_samples_split": 5  
        })
        model = RandomForestRegressor(**params)
        return model

    def _define_gbr(self):
        params = dict({
            "n_estimators": 50,
            "max_leaf_nodes": 4,
            "max_depth": None,
            "random_state": 2,
            "min_samples_split": 5,
            "learning_rate": 0.1,
            "subsample": 0.5
        })
        model = GradientBoostingRegressor(**params)
        return model
    
    def _define_regressor(self):
        if self.model_name == "RF":
            model = self._define_rf()
        elif self.model_name == "GBR":
            model = self._define_gbr()
        return model

    def train(self, train_x: np.ndarray, train_y: np.ndarray):
        self.model = self._define_regressor()
        self.model.fit(train_x, train_y)
        with open(f'trained_models/{self.model_name}.pkl','wb') as f:
            pickle.dump(self.model, f)

    def predict(self, pred_x: np.ndarray):
        with open(f'trained_models/{self.model_name}.pkl', 'rb') as f:
            self.model = pickle.load(f)
        pred_val = self.model.predict(pred_x)
        return pred_val
