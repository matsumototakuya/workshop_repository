# ライブラリのimport
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# データの読み込み
dataset = pd.read_excel('basutienn.xlsx')

# 教師データとテストデータに分割
train_data, test_data, train_target, test_target = train_test_split(dataset.iloc[:, 1:5], dataset.iloc[:, 5],
                                                                    test_size=0.3, random_state=0)

rg = RFR(n_jobs=1, random_state=0, n_estimators=5)  # randomforest

rg.fit(train_data, train_target)
pred = rg.predict(test_data)

# 学習済みモデルの保存
joblib.dump(rg, "rf.pkl", compress=True)

# 予測精度
print("result: ", rg.score(test_data, test_target))

# データの保存
data = dataset.iloc[:, 1:5].values
target = dataset.iloc[:, 5].values
np.save("data", data)
np.save("target", target)
