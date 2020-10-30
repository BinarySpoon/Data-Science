# libraries used -->
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score
from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# models -->
from sklearn import datasets, linear_model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures


# loading data -->
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston_data = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)

# Defining Target label column -->
boston_data.rename({"MEDV":"Target"},axis='columns', inplace=True)

# Preprocessing -->
#print(boston_data.isnull().sum())

# distribution box plots -->
fig, axs = plt.subplots(ncols=7,nrows=2,figsize=(20,10))
index = 0
axs = axs.flatten()
for k,v in boston_data.items():
    sns.boxplot(y=k, data=boston_data, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# Outlier percentage -->
for k ,v in boston_data.items():
    quarter_1 = v.quantile(0.25)
    quarter_3 = v.quantile(0.75)
    inter_quartile = quarter_3 - quarter_1
    value_columns = v[(v <= quarter_1 - 1.5*inter_quartile) | (v >= quarter_3 + 1.5*inter_quartile)]
    percentage = np.shape(value_columns)[0]*100.0 / np.shape(boston_data)[0]

    # Print out result -->
    #print("Column {} outliers = {:.2f}".format(k,percentage))


# Distribution plot -->
fig, axis = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index= 0
axis = axis.flatten()
for k,v in boston_data.items():
    sns.distplot(v, ax=axis[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad = 5.0)

# Distribution plot {Target Column} -->
sns.distplot(boston_data['Target'], bins=40)

# removing target outliers -->
boston_data = boston_data[~(boston_data['Target'] >= 50.0)]


# Heat map -->
plt.figure(figsize=(20,5))
sns.heatmap(boston_data.corr().abs(), annot=True)

# Correlation Curves -->
features = ['NOX','PTRATIO','RM','TAX','LSTAT','INDUS']
min_max_scaler = preprocessing.MinMaxScaler()
x_data = boston_data.loc[:,features]
y_label = boston_data['Target']
x_data = pd.DataFrame(data=min_max_scaler.fit_transform(x_data), columns = features)
fig, axis = plt.subplots(ncols=3, nrows=2, figsize=(20,10))
index = 0
axis = axis.flatten()
for i,col in enumerate(features):
    sns.regplot(y=y_label,x=x_data[col], ax=axis[i])
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=5.0)

# Printout results -->
plt.plot()
plt.show()
