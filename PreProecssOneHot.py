from sklearn.preprocessing import OneHotEncoder
import pandas as pd
data = pd.read_excel(r"C:\files\repo\ami\data\all.xlsx",sheet_name=0)  # 读取原始Excel文件
X = data.iloc[:,3:5]   # 将Excel中第4列提取出来（即非数值变量）
result = OneHotEncoder(categories = 'auto').fit_transform(X).toarray()
print(result)
result1 = pd.DataFrame(result)
result1.to_csv(r'C:\files\repo\ami\data\test1.csv',header=None,index=None)