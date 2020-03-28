import pandas as pd
# pd.options.mode.chained_assignment = None
import numpy as np
import lightgbm as lgb


from sklearn import preprocessing
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

# criteria
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,recall_score, precision_score, roc_curve




# 加载数据
data = pd.read_csv('./data/train_data_modified3.csv', encoding='utf-8')
data = data.sample(frac=1, random_state=42)
data_x = data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]]
lbl = preprocessing.LabelEncoder()
data_x['M'] = lbl.fit_transform(data_x['M'].astype(str))#将提示的包含错误数据类型这一列进行转换
data_x['EMERGENCY'] = lbl.fit_transform(data_x['EMERGENCY'].astype(str))#将提示的包含错误数据类型这一列进行转换
data_y = data.iloc[:,[0]]

# 准备一个train/test来构建模型。
x_train, x_test, y_train, y_test = train_test_split(data_x,
                                                    data_y, 
                                                    test_size=0.2, 
                                                    random_state=52,
                                                    )

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

feature_ami = ['gender','admission','ICD9_CODE','Age','GCS_min','Urine_max','Urine_min','Urine_mean','PaO2_mean','Abnormal_HR_P','WBC_max','WBC_min','Tep_max','Tep_min','Tep_range','Tep_var','USBP_max','USBP_min','USBP_range','USBP_var','HR_range','HR_max','HR_min','HR_var','Bil_max','Bil_min','K_max','K_min','Na_max','Na_min','urea_max','urea_min','SBL_max','SBL_min','SBP_max','SBP_min','SBP_range','SBP_var','creatinine']

gbm = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=5, max_depth=5, learning_rate=0.03, n_estimators=400,feature_fraction=0.9,min_data_in_leaf=4)
# gbm.fit(x_train, y_train, feature_name=feature_ami,categorical_feature=['gender','admission'])
gbm.fit(x_train, y_train, categorical_feature=[1,2])

y_pred_gbm = gbm.predict(x_test)


y_pred_gbm_pr = gbm.predict_proba(x_test)[:,1]
fpr_gbm,tpr_gbm,thresholds  = roc_curve(y_test,y_pred_gbm_pr)


# 评价指标
print("auc面积:",roc_auc_score(y_test, y_pred_gbm_pr))
print("精确率:",precision_score(y_test, y_pred_gbm))
print("召回率:",recall_score(y_test, y_pred_gbm))
print("正确率:",accuracy_score(y_test, y_pred_gbm))
print("F1值:",f1_score(y_test, y_pred_gbm))