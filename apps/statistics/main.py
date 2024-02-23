import matplotlib.pyplot as plt
import  numpy as np
import pandas as pd
import seaborn as sns
import researchpy as  rp
import statsmodels.stats.api as sms
from scipy.stats import bernoulli
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import norm
import scipy.stats as stats
import pylab
from scipy.stats import shapiro
from  statsmodels.stats.descriptivestats import sign_test
from  statsmodels.stats.proportion import proportions_ztest



"""
#sampling technique

population = np.random.randint(0,80,10000)

np.random.seed(101)

sam_pop = np.random.choice(a = population,size=100)

print(sam_pop.mean())
print(population.mean())


#Data Summary

tips = sns.load_dataset("tips")

df_tips = tips.copy()

print(df_tips.head())
print(df_tips.shape)
print(df_tips.info)
print(df_tips.nunique())
print(df_tips.describe().T)
print(df_tips[["tip","total_bill"]].cov())
print(df_tips[["tip","total_bill"]].corr())


#confidence interval

prices = np.random.randint(10,110,1000)

print(prices.mean())

print(sms.DescrStatsW(prices).tconfint_mean())

#bernoulli

p = 0.6
rv = bernoulli(p)

print(rv.pmf(k=1))
print(rv.pmf(k=0))

#binom

p = 0.01
n = 100
rv = binom(n,p)

print(rv.pmf(1))
print(rv.pmf(5))

#poisson

lambda_ = 0.1

rv = poisson(mu= lambda_)

print(rv.pmf(k=0))
print(rv.pmf(k=3))



#Normal Distribution

print(1-norm.cdf(90,80,5))


#stats

tests = np.array([44,44,566,75,12,32,5,676,8,55,100,45,6,77,88,2113,44,56,66,45,45,47,36])

print(stats.describe(tests))

#pd.DataFrame(tests).plot.hist()
#plt.show()

#stats.probplot(tests,dist="norm",plot=pylab)
#pylab.show()

print(shapiro(tests))

print(stats.ttest_1samp(tests,popmean=45))

print(sign_test(tests,300))


count = 40
nobs = 500
value = 0.125

print(proportions_ztest(count,nobs,value))

"""

scs = np.array([300,250])
obser = np.array([1000,1100])

print(proportions_ztest(count=scs,nobs=obser))












