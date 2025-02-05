import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from data_process.krig_data import KrigTrainTestData
from model.tree import TrainTestTree

if __name__ == "__main__":
    cols = ["genHour", "eff_krig", "TA", "RN", "WS", "WD", "HM", "PV", "TD", 
            "PA", "PS", "SS", "SI", "SD_TOT", "SD_HR3", "CA_TOT", "CA_MID", 
            "VS", "TS", "SW038", "WV063", "WV069", "WV073", "IR087", "IR096", 
            "IR105", "IR112", "IR123", "IR133"]
    
    krig_data = KrigTrainTestData(cols)
    model_name = "GBR"

    set_acc, set_day_acc = {}, {}
    set_acc_rfe, set_day_acc_rfe = {}, {}

    # 🔹 Feature Importance 저장을 위한 딕셔너리
    total_feature_importance = {col: 0 for col in cols}
    num_sets = len(krig_data.all_sets)  # 세트 개수

    for set_name in krig_data.all_sets:
        # 데이터 로드
        train_input, train_label, valid_input, valid_label, valid_df = krig_data.read_set_data(set_name)

        # 1️⃣ 기본 GBR 학습
        model_train_test = TrainTestTree(model_name)
        model_train_test.train(train_input, train_label)
        valid_pred = model_train_test.predict(valid_input)
        
        # 2️⃣ Feature Importance 저장
        importance = model_train_test.model.feature_importances_
        for col, imp in zip(cols, importance):
            total_feature_importance[col] += imp  # 중요도 누적

        # 3️⃣ RFE 적용 (상위 15개 특징만 선택)
        selector = RFE(model_train_test.model, n_features_to_select=15, step=1)
        selector.fit(train_input, train_label)

        selected_features = np.array(cols)[selector.support_]

        # 4️⃣ RFE 적용 후 데이터 변환
        train_input_rfe = selector.transform(train_input)
        valid_input_rfe = selector.transform(valid_input)

        # 5️⃣ RFE 적용 후 GBR 재학습
        model_train_test_rfe = TrainTestTree(model_name)
        model_train_test_rfe.train(train_input_rfe, train_label)
        valid_pred_rfe = model_train_test_rfe.predict(valid_input_rfe)

        # 6️⃣ 성능 평가 (R² Score)
        day_label = valid_label[(valid_df["genHour"] >= 10) & (valid_df["genHour"] <= 16)]
        day_pred = valid_pred[(valid_df["genHour"] >= 10) & (valid_df["genHour"] <= 16)]
        set_acc[set_name] = r2_score(valid_label, valid_pred)
        set_day_acc[set_name] = r2_score(day_label, day_pred)

        day_pred_rfe = valid_pred_rfe[(valid_df["genHour"] >= 10) & (valid_df["genHour"] <= 16)]
        set_acc_rfe[set_name] = r2_score(valid_label, valid_pred_rfe)
        set_day_acc_rfe[set_name] = r2_score(day_label, day_pred_rfe)

    # 🔹 모든 세트에서의 평균 Feature Importance 계산
    avg_feature_importance = {col: total_feature_importance[col] / num_sets for col in cols}
    feature_importance_df = pd.DataFrame(list(avg_feature_importance.items()), columns=["Feature", "Importance"])
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # 🔹 Feature Importance 출력 (전체 세트 평균)
    print("\n📊 전체 데이터셋 평균 Feature Importance:")
    print(feature_importance_df)

    print("📊 기본 GBR 평균 R² Score:", round(np.mean(list(set_acc.values())), 3))
    print("📊 기본 GBR (10시~16시) 평균 R² Score:", round(np.mean(list(set_day_acc.values())), 3))
    print("📊 RFE 적용 후 GBR 평균 R² Score:", round(np.mean(list(set_acc_rfe.values())), 3))
    print("📊 RFE 적용 후 GBR (10시~16시) 평균 R² Score:", round(np.mean(list(set_day_acc_rfe.values())), 3))
