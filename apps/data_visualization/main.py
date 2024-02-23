import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pr

plantes = sns.load_dataset("planets")
plantes_copy = plantes.copy()

"""
#1)examine
print(planetes)

print(planetes.head())
print(planetes.tail())

print(planetes_copy.info())

planetes_copy.method = pd.Categorical(planetes.method)

print(planetes_copy.info())

print(planetes_copy.describe().T)

print(planetes_copy.isnull().values.any())

print(planetes_copy.isnull().sum())



print(planetes_copy.isnull().sum())

planetes_copy["orbital_period"] = planetes_copy["orbital_period"].fillna(planetes_copy["orbital_period"].mean())

print(planetes_copy["orbital_period"].mode())
print(planetes_copy["orbital_period"].count())

planetes_copy.fillna(planetes_copy.mean(), inplace=True)


#2)categorical variable examine

cat_planetes = plantes_copy.select_dtypes(include=["object"])

print(cat_planetes)

print(cat_planetes.head(10))
print(cat_planetes.tail(10))

print(cat_planetes.method.unique())

print(cat_planetes["method"].value_counts())

print(cat_planetes["method"].value_counts().count())

cat_planetes["method"].value_counts().plot.barh()

plt.show()

#3)numerical veriable examine

num_planetes = plantes.select_dtypes(include=["float64","int64"])

print(num_planetes.head())

print(num_planetes.describe().T)

print(num_planetes["mass"].describe().T)

"""
"""
#4)Graphics
# -------
#4.1)Bar-plot

diamonds = sns.load_dataset("diamonds")

dm = diamonds.copy()

print(dm.head())
print(dm.tail())
print(dm.describe())
print(dm.value_counts())

from  pandas.api.types import CategoricalDtype

cut_dm = dm.cut.astype(CategoricalDtype(ordered=True))

print(cut_dm)

cut_catgr = ["Fair","Good","Very Good","Premium","Ideal"]

diamonds.cut = diamonds.cut.astype(CategoricalDtype(categories=cut_catgr,ordered=True))

print(diamonds.cut)
"""
#diamonds.cut.value_counts().plot.barh().set_title("Diamonds Cut Bar chart")

#plt.show()

#sns.barplot(x="cut",y=diamonds.cut.index,data= diamonds)

#plt.title("Diamonds Cut Bar chart")

#plt.show()

#sns.catplot(x= "cut",y="price",data=diamonds)

#plt.title("Diamonds Cut-Price Bar chart")

#plt.show()

#sns.catplot(x="cut",y="price",hue="color",data=diamonds)

#plt.show()

#sns.barplot(x="cut",y="price",hue="color",data=diamonds)
#plt.title("Diamonds Price-cut,color Bar chart")
#plt.show()

#4.2)dis-plot,cat-plot

#sns.displot(diamonds.price,bins =70, kde=False)
#plt.title("Diamonds Price-Count,color Bar chart")
#plt.show()

#sns.kdeplot(diamonds.price,shade=True)
#plt.title("Diamonds Price-Count Chart")
#plt.show()
"""
sns.FacetGrid(diamonds,
              hue= "cut",
              height=5,
              xlim=(0.10000)
              ).map(sns.kdeplot,"price",shade= True).add_legend()

plt.show()


sns.catplot(x="cut",y="price",hue="color",kind="point",data=diamonds)
plt.title("Diamonds Price,color,point Chart")
plt.show()

"""

"""
#4.3)box-plot

tips = sns.load_dataset("tips")

print(tips.head())
print(tips.describe().T)

print(tips.sex.value_counts())
print(tips.day.value_counts())
print(tips.smoker.value_counts())
print(tips.time.value_counts())
"""
"""
sns.boxplot(x= tips["total_bill"])
plt.show()

sns.boxplot(x= tips["total_bill"],orient="v")
plt.show()

sns.boxplot(x="day",y="total_bill",data=tips)
plt.title("Price-Total Bill Box Plot")
plt.show()


sns.boxplot(x="time",y="total_bill",data=tips)
plt.title("Price-Time Box Plot")
plt.show()

sns.boxplot(x="size",y="total_bill",data=tips)
plt.title("Size-Time Box Plot")
plt.show()

sns.boxplot(x="size",y="total_bill",hue = "sex",data=tips)
plt.title("Size-Time-Sex Box Plot")
plt.show()

#4.4)Violin-plot

sns.catplot(y="total_bill",kind="violin",data=tips)
plt.show()

sns.catplot(x= "day",y="total_bill",hue="sex",kind="violin",data=tips)
plt.title("Day-Total bill-Sex violin Plot")
plt.show()

#4.5)Scatter-plot

sns.scatterplot(x="total_bill",y="tip",data=tips)
plt.title("Tip-Total bill-Scatter Plot")
plt.show()

sns.scatterplot(x="total_bill",y="tip",hue="day",style="time",data=tips)
plt.title("Tip-Total bill-day-time Scatter Plot")
plt.show()

sns.lmplot(x="total_bill",y="tip",hue="day",col="time", row="sex",data=tips)
plt.title("Tip-Total bill-day-time-sex Scatter Plots")
plt.show()

#4.6)pair-plot
iris = sns.load_dataset("iris")

print(iris.head())
print(iris.shape)

sns.pairplot(iris,hue="species",markers=["o","s","D"],kind="reg")
plt.show()
"""
"""
#4.7)heat map

flights = sns.load_dataset("flights")
flt = flights.copy()

print(flights.head())
print(flights.describe().T)
print(flights.shape)

flt = flt.pivot(index="month", columns="year", values="passengers")
print(flt)

sns.heatmap(flt,annot=True, fmt="d", linewidths=0.6)
plt.show()

#4.8)line-plot
fmri = sns.load_dataset("fmri")

print(fmri.head())
print(fmri.shape)

print(fmri.event.describe())
print(fmri.timepoint.describe())


sns.lineplot(x="timepoint",y="signal",
                                    hue = "event",style="event",
                                                                markers=True, dashes=False,data=fmri)
plt.show()
"""
#4.9)

import yfinance as yf

yf = yf.download("AAPL", start="2016-01-01", end="2019-08-25")
print(yf.head())
print(yf.shape)

close = yf.Close
print(close)

#close.plot()
#plt.show()

print(close.index)




  