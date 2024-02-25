import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import missingno as msno
from ycimpute.imputer import knnimput
from ycimpute.imputer import iterforest
from ycimpute.imputer import EM
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

"""
diamonds = sns.load_dataset("diamonds")
diamonds = diamonds.select_dtypes(include=["float64","int64"])
diamonds = diamonds.dropna()

print(diamonds.head())
print(diamonds.tail())
print(diamonds.describe())
dmd_table = diamonds["table"]
print(dmd_table.head())
Q1 = dmd_table.quantile(0.25)
Q3 = dmd_table.quantile(0.75)
IQR = Q3-Q1

lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR

print(lower_bound)
print(upper_bound)

outlier_table = (dmd_table < lower_bound) | (dmd_table > upper_bound)

print(outlier_table)

#sns.boxplot(x = dmd_table)
#plt.show()


print(dmd_table[outlier_table])

print(dmd_table[outlier_table].index)

dmd_table = pd.DataFrame(dmd_table)

print(dmd_table.shape)



#outlier analysis
#1) clear method


clean_dmd_table = dmd_table[~((dmd_table < lower_bound) | (dmd_table > upper_bound)).any(axis = 1)]

print(clean_dmd_table.shape)


#2)mean method

print(dmd_table.mean())

dmd_table[outlier_table] = dmd_table.mean()

print(dmd_table[outlier_table])


#3)suppression method

dmd_table[outlier_table] = lower_bound
dmd_table[outlier_table] = upper_bound


#4)local outlier factor

clf =LocalOutlierFactor(n_neighbors=20,contamination=0.1)
clf.fit_predict(diamonds)
diamonds_scores = clf.negative_outlier_factor_

print(diamonds_scores[:10])

np.sort(diamonds_scores)

print(diamonds_scores)

threshold_value = np.sort(diamonds_scores)[13]

outlier_values = diamonds_scores > threshold_value

clean_diamond = diamonds[~outlier_values]

print(diamonds[diamonds_scores == threshold_value])

"""
"""
#missing data

V1 = np.array([1,3,5,np.NaN,9,np.NaN,13,15,17])
V2 = np.array([0,2,np.NaN,np.NaN,8,10,12,14,np.NaN])
V3 = np.array([np.NaN,np.NaN,5,7,13,17,19,23,np.NaN])

df = pd.DataFrame({"V1": V1,"V2": V2,"V3":V3})

print(df)
print(df.isnull())
print(df.isnull().sum())
print(df.notnull().sum())
print(df.isnull().sum().sum())
print(df.isnull().sum())
print(df.notnull().sum().sum())
print(df[df.isnull().any(axis =1)])
print(df[df.notnull().all(axis =1)])


#clear method

df_clean = df.copy()

df_clean = df_clean.dropna()

print(df_clean)


#1)Data Imputation

df_imp = df.copy()


df_imp["V1"] = df_imp["V1"].fillna(df_imp["V1"].mean())

print(df_imp["V1"])

df_imp["V2"] = df_imp["V2"].fillna(df_imp["V2"].mean())
print(df_imp["V2"])

df_imp["V3"] = df_imp["V3"].fillna(0)
print(df_imp["V3"])

df_mean = df.apply(lambda x: x.fillna(x.mean()),axis=0)

print(df_mean)

#visualization of missing data

#msno.bar(df)
#plt.show()

#msno.matrix(df)
#plt.show()

df_planets = sns.load_dataset("planets")

print(df_planets.head())

print(df_planets.isnull().sum())

#msno.matrix(df_planets)
#plt.show()

#msno.heatmap(df_planets)
#plt.show()



#Imputation of missing data

print(df["V1"].fillna(0))
print(df["V2"].fillna(df["V2"].mean()))

print(df.apply(lambda x : x.fillna(x.mean(),axis=0)))

print(df.fillna(df.mean()[:]))

print(df.where(pd.notna(df),df.mean(),axis="colomns"))


#Imputation of categorical missing data

V1 = np.array([1,3,5,np.NaN,9,np.NaN,13,15,17])
V2 = np.array([0,2,np.NaN,np.NaN,8,10,12,14,np.NaN])
V3 = np.array([np.NaN,np.NaN,5,7,13,17,19,23,np.NaN])
V4 = np.array(["IT","IT","IT","IK","IK","IT","IK","IK","IT"])
df = pd.DataFrame({"V1": V1,"V2": V2,"wage":V3,"department":V4})

print(df)
print(df.groupby("department")["wage"].mean())

print(df["wage"].fillna(df.groupby("department")["wage"].transform("mean")))

"""


df_titanic = sns.load_dataset("titanic")
df_titanic = df_titanic.select_dtypes(include=["float64","int64"])
print(df_titanic.head())
print(df_titanic.isnull().sum())

var_names = list(df_titanic)

np_titanic = np.array(df_titanic)


"""
#knn

df_knn_titanic = knnimput.KNN(k=4).complete(np_titanic)

df_knn_titanic = pd.DataFrame(df_knn_titanic,columns=var_names)

print(df_knn_titanic.head())
print(df_knn_titanic.describe().T)
"""
"""
#random forest

df_rf_titanic = iterforest.IterImput().complete(np_titanic)

df_rf_titanic = pd.DataFrame(df_rf_titanic,columns=var_names)


print(df_rf_titanic.head())
print(df_rf_titanic.describe().T)
"""
"""
#EM

df_em_titanic =  EM.complete(np_titanic)
df_em_titanic = pd.DataFrame(df_em_titanic,columns=var_names)


#variable standardization

V1 = np.array([1,3,5,7,9,11,13,15,17])
V2 = np.array([0,2,4,6,8,10,12,14,16])
V3 = np.array([23,5,7,13,17,19,23,29])

df = pd.DataFrame({"V1": V1,"V2": V2,"V3":V3})

df = df.asytpe(float)

print(preprocessing.scale(df))
print(preprocessing.normalize(df))
scaler = preprocessing.MinMaxScaler(feature_range=(0,100))
print(scaler.fit_transform(df))
"""
"""
#variable transformation

df_tips = sns.load_dataset("tips")

lbe = LabelEncoder()

print(lbe.fit_transform(df_tips["sex"]))

df["new_day"] = np.where(df["day"].str.contanis("Sunday"),1,0)

print(df)

print(lbe.fit_transform(df["day"]))

df_one_hot = pd.get_dummies(df,columns=["sex"],prefix=["sex"]

print(df_one_hot.head())
"""
#variable transformation"


V1 = np.array([1,3,5,7,9,11,13,15,17])
V2 = np.array([0,2,4,6,8,10,12,14,16])
V3 = np.array([23,5,7,13,17,19,23,29])

df = pd.DataFrame({"V1": V1,"V2": V2,"V3":V3})

binarizer = preprocessing.Binarizer(threshold=5).fit(df)
print(binarizer.transform(df))


df_tips = sns.load_dataset("tips")

df_tips_copy = df_tips.copy()

df_tips_copy["new_sex"] = df_tips_copy["sex"].cat.codes

print(df_tips_copy)



























