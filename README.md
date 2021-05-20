# Predict the Morality of AMI

## About

using decision-tree based algorithm to predict the morality of acute myocardial infarction

The samples are arrived from the MIMIC-Ⅲ database，and 4788 patients are selected, according to ICD-9 code from 41000 to 41091 (it means patents are diagnosed of acute myocardial infarction)

XGboost, LightGBM and GBDT perform better than others, the accuracy reaches more than 87%

## 基于机器学习的急性心肌梗塞死亡率预测

用MIMIC数据库的数据作为训练集的死亡率预测，用于预测

当年本科水毕设的灌水项目，实际上没卵用。因为这个MIMIC数据库里面的东西，都是一些离散的检查指标（每天一次或者几小时一次），然后跟心肌梗死有关的都是些血压之类的东西。，而病人都重症监护了，一般会有心电监护仪之类的东西，然后直接看心电图，准确率可以轻易达到95%以上。或者去做彩超和冠脉造影直接拿着片子找医生会诊。

其他一些像肾衰竭等疾病的预测还有一点意义。

~~没想到一年多了还有人想看我这破玩意的处理过程和模型，放出来了~~。**不过，不建议用于除了水论文之外的任何用途。**

当时跑出来效果比较好的是XGBoost和LightGBM一类的boosting方法，当时还用的是MIMIC-III。现在已经出MIMIC-IV，建议自己下载下来重新洗一遍数据构造数据集；据说lstm之类的方法效果也不错，可以试试。
