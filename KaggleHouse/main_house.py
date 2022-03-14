import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score


pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 300)

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# поищем количество столбцов с путсными данными
# print(train_data.isnull().sum())
null_data = train_data.isnull().sum()
null_data = null_data[null_data > 0]
# print(null_data)

# удалим столбцы с пустыми данными по индексам и столбец с id
X = train_data.drop(null_data.index, axis=1)
# X = X.drop(X[X.Id == 1299].index).reset_index(drop=True)
# X = X.drop(X[X.Id == 524].index).reset_index(drop=True)
X = X.drop(['Id', 'SalePrice'], axis=1)

print(X)

X_TEST = test_data.drop(null_data.index, axis=1)
X_TEST = X_TEST.drop(['Id'], axis=1)

y = train_data.SalePrice
# print(X.head())

# заменим не числовые значения на числовые
X = pd.get_dummies(X)
X_TEST = pd.get_dummies(X_TEST).fillna(0)
for col in (list(X.columns)):
    if col not in (list(X_TEST.columns)):
        X = X.drop([col], axis=1)

for col in (list(X_TEST.columns)):
    if col not in (list(X.columns)):
        X_TEST = X_TEST.drop([col], axis=1)

# print(len(list(X_TEST.columns)))
# print(len(list(X.columns)))
# print(x_TEST.columns)
# print('---------------')
# print(X.columns)

#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# # param_rf = {'n_estimators': range(10, 91, 10), 'max_depth': range(1, 24, 2),
# #                 'min_samples_split': range(2, 16, 2), 'min_samples_leaf': range(1, 18), 'criterion': ['gini', 'entropy']}
# param_rf = {'criterion': ['entropy'], 'max_depth': [12], 'min_samples_leaf': [12], 'min_samples_split': [6], 'n_estimators': [30]}
# clf_rf = RandomForestClassifier()
# grid_search_cv_clf = GridSearchCV(clf_rf, param_rf, cv=5, n_jobs=-1)
# grid_search_cv_clf.fit(X_train, y_train)
# best_clf = grid_search_cv_clf.best_estimator_

# print(r2_score(y_test, best_clf.predict(X_test)))


# print(grid_search_cv_clf.best_params_)
# print(r2_score(y_test, best_clf.predict(X_test)))
# res = best_clf.predict(X_TEST)
# res = list(res)
# print(best_clf.score(X_test, y_test))
# print(res)

# 390
# 0.8405999695935691________{'n_estimators': 390}

# max = 0
# best_param = 0
# for number in range(380, 1001, 10):
#     param_rf = {'n_estimators': [number]}
#     # param_rf = {'n_estimators': [number]}
#     clf_rf = RandomForestClassifier()
#     grid_search_cv_clf = GridSearchCV(clf_rf, param_rf, cv=5, n_jobs=-1)
#     grid_search_cv_clf.fit(X_train, y_train)
#     best_clf = grid_search_cv_clf.best_estimator_
#     score = r2_score(y_test, best_clf.predict(X_test))
#     print(number)
#     if score > max:
#         max = score
#         best_param = grid_search_cv_clf.best_params_
#         print(f'{max}________{best_param}')

from sklearn.model_selection import RandomizedSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

param_rf = {'n_estimators': [200],
                            'criterion': ['entropy'], 'bootstrap': [True],
                            'max_features': ['log2']}
                # param_rf = {'n_estimators': [number]}
clf_rf = RandomForestClassifier()
                # grid_search_cv_clf = GridSearchCV(clf_rf, param_rf, cv=5, n_jobs=-1)
                # grid_search_cv_clf.fit(X_train, y_train)
rs = RandomizedSearchCV(clf_rf, param_rf, n_jobs=-1, cv=5,
                                        n_iter=10, verbose=1, random_state=42)
rs.fit(X_train, y_train)
best_clf = rs.best_estimator_
score = best_clf.score(X_test, y_test)

res = best_clf.predict(X_TEST)
res = list(res)


# max = 0
# best_param = 0
# for number in range(20, 211, 10):
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#     for max_depth in range(4, 24, 2):
#         for min_samples_split in range(2, 16, 2):
#             for min_samples_leaf in range(1, 15):
#                 param_rf = {'n_estimators': [number], 'max_depth': [max_depth],
#                             'min_samples_split': [min_samples_split], 'min_samples_leaf': [min_samples_leaf],
#                             'criterion': ['entropy'], 'bootstrap': [True, False],
#                             'max_features': ['log2', 'sqrt', None],
#                             'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int))}
#                 # param_rf = {'n_estimators': [number]}
#                 clf_rf = RandomForestClassifier()
#                 # grid_search_cv_clf = GridSearchCV(clf_rf, param_rf, cv=5, n_jobs=-1)
#                 # grid_search_cv_clf.fit(X_train, y_train)
#                 rs = RandomizedSearchCV(clf_rf, param_rf, n_jobs=-1, cv=5,
#                                         n_iter=10, verbose=1, random_state=42)
#                 rs.fit(X_train, y_train)
#                 best_clf = rs.best_estimator_
#                 score = best_clf.score(X_test, y_test)
#                 print(number)
#                 if score > max:
#                     max = score
#                     best_param = rs.best_params_
#                 print(f'{max}________{best_param}')

#
#
# print(best_param, max)
    # print(grid_search_cv_clf.best_params_)
    # print(f'{r2_score(y_test, best_clf.predict(X_test))} ________ {number}________{grid_search_cv_clf.best_params_}')

# {'max_depth': 17, 'min_samples_leaf': 3, 'min_samples_split': 12, 'n_estimators': 40} 0.7950924569743293
# 0.8297932396492447________{'max_depth': 17, 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 140}

list_id = range(1461, 2920)
# print(len(list_id))
res_data = pd.DataFrame({'Id': list_id, 'SalePrice': res})
filename = "sample_submission.csv"
# # print(res_data)
res_data.to_csv(filename, sep=',', index=False)
