import pandas as pd
import numpy as np
import seaborn as sns

"""
#Series
seri = pd.Series([1,2,3,4,5])

print(seri)

print(seri.values)

print(seri.head(2))

print(seri.tail(2))

seri1= pd.Series([6,7,8,9,0],index=[6,7,8,9,"sıfır"])

print(seri1)

seri2 = pd.Series({"a": 1,"b":2,"c":3})

print(seri2)

print(pd.concat([seri,seri1]))


arr = np.array([1,3,5,7,9])

seri1 = pd.Series(arr)


#DataFrame

liste1 = [1,2,3,4,5]

print(pd.DataFrame(liste1,columns=["Liste1"]))

arr = np.arange(1,13).reshape((4,3))

df = pd.DataFrame(arr,columns=["ilk","orta","son"])

print(df)

print(df.columns)

df.columns = (1,2,3)

print(df)

print(df.axes)

arr1 = np.random.randint(100, size = 5)

arr2 = np.random.randint(100, size = 5)

arr3 = np.random.randint(100, size = 5)

sozluk = {"key1": arr1, "key2": arr2,"key3":arr3}

df1 = pd.DataFrame(sozluk)

print(df1)

print(df.values)

print(df1.keys)

print(df1.index)

df1.index = ["a","b","c","d","e"]

print(df1)

df1.drop("a",axis=0, inplace=True)

df1.drop("key1",axis=1,inplace=True)

print(df1["key2"])

df1["key4"] = df1["key2"] * df1["key3"]


#loc ve iloc

arr = np.random.randint(1,101, size= (5,5))

df = pd.DataFrame(arr,columns=["a","b","c","d","e"])

print(df["a"])

print(df.loc[0:3])

print(df.iloc[0:3])

print(df.loc[:2,:"c"])

print(df.iloc[:2,:3])

print(df[:2][["b","e"]])

#join

arr1 = np.random.randint(1,20, size= (4,3))

df1 = pd.DataFrame(arr1,columns=["ilk","orta","son"])

df2 = df1+80

print(pd.concat([df1,df2]))


print(pd.concat([df1,df2] , ignore_index= True))


df2.columns = ["ilk","ortanca","son"]


print(pd.concat([df1,df2], join="inner"))



df1 = pd.DataFrame({"çalısanlar": ["Murat","Gamze","Kerem","Berkan"],
                    "meslekler":  ["ogretmen","polis","savcı","asker"] })

print(df1)

df2 = pd.DataFrame({"çalısanlar": ["Murat","Gamze","Kerem","Berkan"],
                    "giris_yılı":  [2012,2014,2020,2024] })


print(df2)

df3 = pd.merge(df1,df2)

print(df3)

# işlemler 

df = sns.load_dataset("planets")

print(df.mean())

print(df["year"].mean())

print(df["year"].count())

print(df["year"].min())

print(df["year"].max())

print(df["year"].sum())

print(df["year"].std())

print(df["year"].var())

print(df.describe())

print(df.describe().T)



#gruplandırma

df1 = pd.DataFrame({"gruplar": ["a","b","c","d","a","b","c"],
                    "notlar":  [35,49,42,65,23,57,89] })

df1.groupby("gruplar")

print(df1)

print(df1.groupby("gruplar").mean())



#aggregate, filter,apply

df= pd.DataFrame({"sınıflar": ["a","b","c","a","b"],
                   "degerler": [10,35,67,33,77],
                    "notlar":[25,37,33,78,99]})

print(df)

print(df.groupby("sınıflar").describe())

print(df.groupby("sınıflar").aggregate([min,np.median,max]))

print(df.groupby("sınıflar").aggregate({"degerler":min,"notlar":max}))

print(df.apply(np.sum))

print(df.groupby("sınıflar").apply(np.sum))


#pivot_table
titanic = sns.load_dataset("titanic")

print(titanic)
print(titanic.head())

print(titanic.groupby("sex")["survived"].mean())

print(titanic.groupby(["sex","class"])[["survived"]].aggregate("mean"))

print(titanic.pivot_table("survived",index="sex",columns="class"))

"""
#Dosya okuma

print(pd.read_csv("reading_data/ornekcsv.csv",sep=";"))


print(pd.read_csv("reading_data/duz_metin.txt"))


print(pd.read_excel("reading_data/ornekx.xlsx"))

























































