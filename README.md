# Predict the Morality of AMI
using decision-tree based algorithm to predict the morality of acute myocardial infarction

The samples are arrived from the MIMIC-Ⅲ database，and 4788 patients are selected, according to ICD-9 code from 41000 to 41091 (it means patents are diagnosed of acute myocardial infarction)

XGboost, LightGBM and GBDT perform better than others, the accuracy reaches more than 87%
