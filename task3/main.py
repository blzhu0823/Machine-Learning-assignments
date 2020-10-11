#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 1:50 下午
# @Author  : zbl
# @Email   : funnyzhu1997@gmail.com
# @File    : main.py
# @Software: PyCharm


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import sklearn.tree as tree


boston = load_boston()
print(boston.keys())
data = boston.data
label = boston.target
print('data shape:', data.shape)
print('label shape:', label.shape)

# plt.figure()
# for i in range(data.shape[-1]):
#     plt.subplot(4, 4, i+1)
#     plt.scatter(data[:, i], label)
#     plt.xlabel(boston.feature_names[i])
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=22)


st_X, st_y = StandardScaler(), StandardScaler()
X_train = st_X.fit_transform(X_train)
X_test = st_X.transform(X_test)
y_train = st_y.fit_transform(y_train.reshape(-1, 1))
y_test = st_y.transform(y_test.reshape(-1, 1))

print('X_train:', X_train.shape)
print('X_test:', X_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
sgdr = SGDRegressor(max_iter=5)
sgdr.fit(X_train, y_train)
sgdr_preds = sgdr.predict(X_test)
print('lr 模型自代评分:', lr.score(X_test, y_test))
print('lr R-score:', r2_score(y_test, lr_preds))
print('lr mean squared:', mean_squared_error(st_y.inverse_transform(y_test), st_y.inverse_transform(lr_preds)))
print('lr mean absolute:', mean_absolute_error(st_y.inverse_transform(y_test), st_y.inverse_transform(lr_preds)))

print()

print('sgdr 模型自代评分:', sgdr.score(X_test, y_test))
print('sgdr R-score:', r2_score(y_test, sgdr_preds))
print('sgdr mean squared:', mean_squared_error(st_y.inverse_transform(y_test), st_y.inverse_transform(sgdr_preds)))
print('sgdr absolute:', mean_absolute_error(st_y.inverse_transform(y_test), st_y.inverse_transform(sgdr_preds)))

print()

ridge_r = Ridge(alpha=1)
ridge_r.fit(X_train, y_train)
ridge_r_preds = ridge_r.predict(X_test)
print('ridge 模型自代评分:', ridge_r.score(X_test, y_test))
print('ridge R-score:', r2_score(y_test, ridge_r_preds))
print('ridge mean squared:', mean_squared_error(st_y.inverse_transform(y_test), st_y.inverse_transform(ridge_r_preds)))
print('ridge mean absolute:', mean_absolute_error(st_y.inverse_transform(y_test), st_y.inverse_transform(ridge_r_preds)))

print()

lasso_r = Lasso(alpha=0.005)
lasso_r.fit(X_train, y_train)
lasso_r_preds = lasso_r.predict(X_test)
print('lasso 模型自代评分:', lasso_r.score(X_test, y_test))
print('lasso R-score:', r2_score(y_test, lasso_r_preds))
print('lasso mean squared:', mean_squared_error(st_y.inverse_transform(y_test), st_y.inverse_transform(lasso_r_preds)))
print('lasso mean absolute:', mean_absolute_error(st_y.inverse_transform(y_test), st_y.inverse_transform(lasso_r_preds)))

print()

dt_r = DecisionTreeRegressor(max_depth=5)
dt_r.fit(X_train, y_train)
dt_r_preds = dt_r.predict(X_test)
print('DecisionTree 模型自代评分:', dt_r.score(X_test, y_test))
print('DecisionTree R-score:', r2_score(y_test, dt_r_preds))
print('DecisionTree mean squared:', mean_squared_error(st_y.inverse_transform(y_test), st_y.inverse_transform(dt_r_preds)))
print('DecisionTree mean absolute:', mean_absolute_error(st_y.inverse_transform(y_test), st_y.inverse_transform(dt_r_preds)))

with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(dt_r, out_file=f)


