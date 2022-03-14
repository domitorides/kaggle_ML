import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 300)

df_train = pd.read_csv('train.csv')

# print(df_train['SalePrice'].describe())

# sns.distplot(df_train['SalePrice'])

# skewness and kurtosis асимметрия и эксцесс
# print("Skewness: %f" % df_train['SalePrice'].skew())
# print("Kurtosis: %f" % df_train['SalePrice'].kurt())



# # Relationship with numerical variables
# # scatter plot grlivarea/saleprice
# # квадратные футы жилой площади над уровнем земли
# # по диаграмме видно, что у них линейная зависимость
# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
#
# # scatter plot totalbsmtsf/saleprice
# # TotalBsmtSF: Total square feet of basement area
# # Общая площадь подвала в квадратных футах
# var = 'TotalBsmtSF'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))



# # Relationship with categorical features
# # box plot overallqual/saleprice
# # Оценивает общий материал и отделку дома.
# var = 'OverallQual'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000)
#
# # box plot YearBuilt/saleprice
# # год постройки.
# var = 'YearBuilt'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# f, ax = plt.subplots(figsize=(16, 8))
# fig = sns.boxplot(x=var, y="SalePrice", data=data)
# fig.axis(ymin=0, ymax=800000)
# plt.xticks(rotation=90)


# В итоге
# Помимо историй, мы можем сделать вывод, что:
#
# «GrLivArea» и «TotalBsmtSF» кажутся линейно связанными с «SalePrice».
# Оба отношения положительны, что означает, что по мере увеличения одной переменной увеличивается и другая.
# В случае TotalBsmtSF мы видим, что наклон линейной зависимости особенно высок.
# «TotalQual» и «YearBuilt» также, похоже, связаны с «SalePrice».
# Взаимосвязь кажется более сильной в случае «OverallQual», где прямоугольная диаграмма показывает,
# как продажные цены растут вместе с общим качеством.


# # Correlation matrix (heatmap style)
# # correlation matrix
# corrmat = df_train.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True)


# # saleprice correlation matrix
# # видем переменные, имеющие наибольший коэф корреляции
# k = 10 # number of variables for heatmap
# corrmat = df_train.corr()
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(df_train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
#                  annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


# посмотрим точечные диаграммы с наибольшей корреляцией
# #scatterplot
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols], size=1.5)


# #missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# # print(missing_data.head(20))
#
# #dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
# print(df_train.isnull().sum().max()) #just checking that there's no missing data missin


# #standardizing data
# saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])
# low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
# high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
# print('outer range (low) of the distribution:')
# print(low_range)
# print('\nouter range (high) of the distribution:')
# print(high_range)

# # bivariate analysis saleprice/grlivarea
# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# deleting points
# удалим 2 странных выброса из пункта bivariate analysis saleprice/grlivarea
df_train.sort_values(by='GrLivArea', ascending=False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
# print(df_train[['Id']])

# # bivariate analysis saleprice/TotalBsmtSF
# var = 'TotalBsmtSF'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# # histogram and normal probability plot
# sns.distplot(df_train['SalePrice'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'], plot=plt)

# applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])

# # transformed histogram and normal probability plot
# sns.distplot(df_train['SalePrice'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['SalePrice'], plot=plt)


# # Done! Let's check what's going on with 'GrLivArea'.
# # histogram and normal probability plot
# sns.distplot(df_train['GrLivArea'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['GrLivArea'], plot=plt)

# data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

# # transformed histogram and normal probability plot
# sns.distplot(df_train['GrLivArea'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['GrLivArea'], plot=plt)

# create column for new variable (one is enough because it's a binary categorical feature)
# if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

# transform data
df_train.loc[df_train['HasBsmt'] == 1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

# # histogram and normal probability plot
# sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)

#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])

plt.show()