
import pandas as pd
# pd.options.mode.chained_assignment = None
import numpy as np
import lightgbm as lgb


from sklearn import preprocessing
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier
# criteria
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,recall_score, precision_score, roc_curve


# 加载数据
data = pd.read_csv('./data/train_data_modified.csv', encoding='utf-8')
data = data.sample(frac=1, random_state=42)
data_x = data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]]
# data_x = data.iloc[:,[1,2,3,4,5,6,7,36]]
lbl = preprocessing.LabelEncoder()
# data_x['M'] = lbl.fit_transform(data_x['M'].astype(str))#将含有字符的类别特征这一列进行转换
# data_x['EMERGENCY'] = lbl.fit_transform(data_x['EMERGENCY'].astype(str))#将含有字符的类别特征这一列进行转换
data_y = data.iloc[:,[0]]

# 准备一个train/test来构建模型。
x_train, x_test, y_train, y_test = train_test_split(data_x,
                                                    data_y, 
                                                    test_size=0.2, 
                                                    random_state=52,
                                                    )

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


from xgboost.sklearn import XGBClassifier
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate =0.09,
    max_depth=4,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=27)
# xgb = HistGradientBoostingClassifier(learning_rate=0.09)
xgb.fit(x_train, y_train)

y_pred_xgb = xgb.predict(x_test)


y_pred_xgb_pr = xgb.predict_proba(x_test)[:,1]
fpr_xgb,tpr_xgb,thresholds  = roc_curve(y_test,y_pred_xgb_pr)

# y_pred_xgb = y_pred_xgb_pr > 0.5
# print(lr.coef_) #W
# print(lr.intercept_) #b
# 评价指标
print("auc面积:",roc_auc_score(y_test, y_pred_xgb_pr))
print("精确率:",precision_score(y_test, y_pred_xgb))
print("召回率:",recall_score(y_test, y_pred_xgb))
print("正确率:",accuracy_score(y_test, y_pred_xgb))
print("F1值:",f1_score(y_test, y_pred_xgb))