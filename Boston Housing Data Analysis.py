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

#
'''
y_label = np.log1p(y_label)
for features in x_data.columns:
    if np.abs(x_data[features].skew()) > 0.5:
        x_data[features] = np.log1p(x_data[features])
'''

# Training Model -->
print('training model')
X_train, X_test, y_train, y_test = train_test_split(x_data,y_label,test_size=0.2,random_state=0)

# Polynomial Linear Regression -->
plr = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge())
plr.fit(X_train, y_train)



# Support Vector Regression -->
svr_rbf = SVR(kernel='rbf',C=1e3,gamma=0.1)
grid_param = {'C':[1e0,1e1,1e2,1e3,1e4,1e5],'gamma':np.logspace(-2,2,5)}
grid_svr = GridSearchCV(svr_rbf, cv=KFold(n_splits=10), param_grid=grid_param)
svr = make_pipeline(StandardScaler(),grid_svr)
svr.fit(X_train,y_train)


# Gradient boosting regressor
gbr = GradientBoostingRegressor(alpha=0.8,learning_rate=0.06,max_depth=2,min_samples_leaf=2,
                                min_samples_split=2, n_estimators=100, random_state=30)
param_grid_gbr = {'n_estimators':[100,200],'learning_rate':[0.1,0.05,0.02,0.005],'max_depth':[2,4,6],
                   'min_samples_leaf':[3,5,9]}
grid_gbr = GridSearchCV(gbr,param_grid=param_grid_gbr)
gbr = make_pipeline(StandardScaler(),grid_gbr)
gbr.fit(X_train,y_train)


# Descision tree regression -->
dtr = DecisionTreeRegressor(max_depth=5)
param_grid_dtr = {'max_depth':[1,2,3,4,5,6,7]}
grid_dtr = GridSearchCV(dtr, cv=KFold(n_splits=10),param_grid=param_grid_dtr)
dtr = make_pipeline(StandardScaler(),grid_dtr)
dtr.fit(X_train,y_train)

#accuracy -->
print('gauging accuracy')
accuracy_plr = plr.score(X_test, y_test)
accuracy_svr = svr.score(X_test,y_test)
accuracy_gbr = gbr.score(X_test,y_test)
accuracy_dtr = dtr.score(X_test,y_test)


# print out results -->
print('Polynomial Linear Regression: {:.2f}'.format(accuracy_plr))
print('Support Vector Regression: {:.2f}'.format(accuracy_svr))
print('Gradient Booster Regression: {:.2f}'.format(accuracy_gbr))
print('Decision Tree Regression: {:.2f}'.format(accuracy_dtr))
plt.plot()
plt.show()
