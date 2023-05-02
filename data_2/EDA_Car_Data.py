import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

df = pd.read_csv('C:/Users/harshitgoel01/Desktop/eda/data_2/data/imports-85.data.txt',names=cols)

df.head()

df = df.replace('?',np.NaN)
df.head()

df.isnull().sum()

df.info()

df['normalized-losses'] = df['normalized-losses'].astype(float)

sns.histplot(data=df['normalized-losses'],kde=True)

sns.swarmplot(x=df['normalized-losses'],data=df)

sns.kdeplot(df['normalized-losses'], shade=True)

sns.boxplot(data=df,x=df['normalized-losses'])

average_normalized_loss = df['normalized-losses'].mean()
average_normalized_loss

average_normalized_loss_median = df['normalized-losses'].median()
average_normalized_loss_median

df['normalized-losses'].replace(np.NaN, average_normalized_loss_median, inplace=True)

df.head()

df['bore'] = df['bore'].astype(float)

average_bore_value = df['bore'].mean()
average_bore_value

median_bore_value = df['bore'].median()
median_bore_value

sns.histplot(data=df['bore'],kde=True)

sns.boxplot(data=df,x=df['bore'])

df['bore'].replace(np.NaN, average_bore_value, inplace=True)

df.isnull().sum()

avg_stroke = df['stroke'].astype(float).mean()
avg_stroke

df['stroke'].replace(np.NaN, avg_stroke,inplace=True)

avg_horsepower = df['horsepower'].astype(float).mean()

df['horsepower'].replace(np.NaN, avg_horsepower,inplace=True)

avg_peak_rpm = df['peak-rpm'].astype(float).mean()

df['peak-rpm'].replace(np.NaN, avg_peak_rpm,inplace=True)

df['num-of-doors'].value_counts()

df['num-of-doors'].value_counts().idxmax()

df['num-of-doors'].replace(np.NaN, 'four',inplace=True)

#price
df.dropna(subset=['price'], axis=0, inplace=True)

df.isnull().sum()

df.dtypes

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df.head()

df.dtypes

df.corr()

sns.regplot(x='engine-size',y='price',data=df)

df.columns

df.dtypes

sns.regplot(x='highway-mpg',y='price',data=df)

sns.regplot(x='city-mpg',y='price',data=df)

sns.regplot(x='peak-rpm',y='price',data=df)

sns.regplot(x='compression-ratio',y='price',data=df)

sns.regplot(x='curb-weight',y='price',data=df)

sns.regplot(x='normalized-losses',y='price',data=df)

sns.regplot(x='wheel-base',y='price',data=df)

sns.regplot(x='length',y='price',data=df)



df['engine-type'].value_counts()

df['make'].value_counts()

df.head()

df.columns

df['aspiration'].value_counts()

sns.boxplot(x='drive-wheels',y='price',data=df)

df.describe(include=['object'])

dumy_df = df[['drive-wheels','price','body-style']]

dumy_df.head()

data_one = df.groupby(['drive-wheels'],as_index=False).mean()
data_one.head()

data = dumy_df.groupby(['body-style','drive-wheels'],as_index=False).mean()

data.head()

grouped_data = data.pivot(index='drive-wheels', columns='body-style')
grouped_data

grouped_data = grouped_data.fillna(0)

grouped_data

sns.heatmap(grouped_data)

from scipy.stats import stats
from scipy.stats import pearsonr

pearson_coef, p_value = pearsonr(df['wheel-base'],df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

t_stat, p_value = stats.ttest_ind(df['wheel-base'],df['price'])

print(p_value)

df.dtypes

pearson_coef, p_value = stats.pearsonr(df['horsepower'].astype('float'), df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 





































