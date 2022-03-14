
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 300)

test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')
# print(titanic_data.head())

# проверим на пустоту наши данные, если много пустых данных, то можно удалить колонку
# print(train_data.isnull().sum())

# print(test_data.isnull().sum())

# далее уберём колонки, которые по логике не несут информации
X = train_data.drop(['label'], axis=1)
y = train_data.label

# print(X[:].sum().to_frame())
# print(test_data[:].sum().to_frame())
# print(check_zero)
# print(X)
#
# pas_id = test_data['PassengerId'].to_list()
x_Test = test_data
# print(X)

sum1 = X[:].sum()
sum2 = test_data[:].sum()
check_zero = pd.DataFrame({'TRAIN': sum1, 'TEST': sum2})
check_zero['sum'] = check_zero['TRAIN'] + check_zero['TEST']

X = X.loc[:, check_zero['sum'] > 300]

x_Test = x_Test.loc[:, check_zero['sum'] > 300]
# print(X)


# param_rf = {'n_estimators': range(10, 51, 10), 'max_depth': range(1, 20, 2),
#             'min_samples_split': range(2, 101, 5), 'min_samples_leaf': range(2, 101, 5)}
# param_rf = {'n_estimators': range(10, 51, 10), 'max_depth': range(1, 13, 2),
#             'min_samples_split': range(2, 10, 2), 'min_samples_leaf': range(1, 8)}




# 0.95285
# получим тестовые данные и обучим по train
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# param_rf = {'max_depth': [12], 'min_samples_leaf': [1], 'min_samples_split': [4], 'n_estimators': [70]}
# clf_rf = RandomForestClassifier()
# grid_search_cv_clf = GridSearchCV(clf_rf, param_rf, cv=5, verbose=1, n_jobs=-1)
# grid_search_cv_clf.fit(X_train, y_train)
# best_clf = grid_search_cv_clf.best_estimator_


# 0.96410
X_train, X_test, y_train, y_test = train_test_split(X, y)
# param_rf = {'n_estimators': range(10, 91, 10), 'max_depth': range(1, 18, 2),
#             'min_samples_split': range(2, 16, 2), 'min_samples_leaf': range(1, 3)}
param_rf = {'max_depth': [17], 'min_samples_leaf': [1], 'min_samples_split': [4], 'n_estimators': [90]}
clf_rf = RandomForestClassifier()
grid_search_cv_clf = GridSearchCV(clf_rf, param_rf, cv=5, verbose=1, n_jobs=-1)
grid_search_cv_clf.fit(X_train, y_train)
best_clf = grid_search_cv_clf.best_estimator_

# lol = [num / 100 for num in range(1, 200, 1)]
# for size in lol:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
#     # param_rf = {'n_estimators': range(10, 91, 10), 'max_depth': range(1, 18, 2),
#     #             'min_samples_split': range(2, 16, 2), 'min_samples_leaf': range(1, 3)}
#     param_rf = {'max_depth': [17], 'min_samples_leaf': [1], 'min_samples_split': [4], 'n_estimators': [90]}
#     clf_rf = RandomForestClassifier()
#     grid_search_cv_clf = GridSearchCV(clf_rf, param_rf, cv=5, verbose=1, n_jobs=-1)
#     grid_search_cv_clf.fit(X_train, y_train)
#     best_clf = grid_search_cv_clf.best_estimator_
#     print(best_clf.score(X_test, y_test), size)



# X_train, X_test, y_train, y_test = train_test_split(X, y)
#     # param_rf = {'n_estimators': range(10, 91, 10), 'max_depth': range(1, 18, 2),
#     #             'min_samples_split': range(2, 16, 2), 'min_samples_leaf': range(1, 3)}
# param_rf = {'max_depth': [23], 'min_samples_leaf': [1], 'min_samples_split': [4], 'n_estimators': [110]}
# clf_rf = RandomForestClassifier()
# grid_search_cv_clf = GridSearchCV(clf_rf, param_rf, cv=5, verbose=1, n_jobs=-1)
# grid_search_cv_clf.fit(X_train, y_train)
# best_clf = grid_search_cv_clf.best_estimator_
# print(best_clf.score(X_test, y_test))





# фичи по важности
# feature_importances = best_clf.feature_importances_
# feature_importances_df = pd.DataFrame({'features': list(X_train), 'feature_importances': feature_importances})
print(grid_search_cv_clf.best_params_)
print(best_clf.score(X_test, y_test))

#
# res = best_clf.predict(x_Test)
# res = list(res)
#
# list_id = range(1, 28001)
# # print(len(list_id))
# res_data = pd.DataFrame({'ImageId': list_id, 'Label': res})
# filename = "sample_submission.csv"
# # # print(res_data)
# res_data.to_csv(filename, sep=',', index=False)