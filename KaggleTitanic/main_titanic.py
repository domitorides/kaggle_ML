from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 300)

titanic_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print(titanic_data.head())

# проверим на пустоту наши данные, если много пустых данных, то можно удалить колонку
# print(titanic_data.isnull().sum())

# print(test_data.isnull().sum())

# далее уберём колонки, которые по логике не несут информации
X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data.Survived
print(X)




pas_id = test_data['PassengerId'].to_list()
x_Test = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# для дерева нужно переконвертировать значения в числовые, поэтому используем get_dummies
# так же нам надо заполнить пропуски в age, используем медиану для этого
X = pd.get_dummies(X)
x_Test = pd.get_dummies(x_Test)
print(X)
exit()

# для пропусков сделаем просто медианый возраст, можно сделать по полу и тестовые данные заполним
X = X.fillna({'Age': X.Age.median()})
x_Test = x_Test.fillna({'Age': x_Test.Age.median()}).fillna(0)

# Обучим
clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf.fit(X, y)

# получим тестовые данные и обучим по train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# clf.fit(X_train, y_train)
#
#
# scores_data = pd.DataFrame()
#
# # сделаем классификаторы с разными глубинами дерева и по кроссвалидации найдём оптимальное значение глубины дерева
# max_depth_value = range(1, 100)
# for max_depth in max_depth_value:
#     clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
#     clf.fit(X_train, y_train)
#     train_score = clf.score(X_train, y_train)
#     test_score = clf.score(X_test, y_test)
#
#     mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=7).mean()
#
#     temp_score = pd.DataFrame({'max_depth': [max_depth], 'train_score': [train_score],
#                                'test_score': [test_score], 'cross_val_score': [mean_cross_val_score]})
#     scores_data = scores_data.append(temp_score)
#
# # смотрим оптимальную глубину
# scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score', 'test_score', 'cross_val_score'],
#                            var_name='set_type', value_name='score')
# print(scores_data_long)
# print(scores_data_long.query("set_type == 'cross_val_score'"))

# # обучаем наилучшее дерево
# best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)
# best_clf.fit(X_train, y_train)
#
# # выводим данные
# res = best_clf.predict(x_Test)
# res = list(res)
#
# res_data = pd.DataFrame({'PassengerId': pas_id, 'Survived': res})
# filename = "gender_submission.csv"
# print(res_data)
# res_data.to_csv(filename, sep=',', index=False)

# 2 способ
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import numpy as np

# clf = tree.DecisionTreeClassifier()
# # parametrs = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30)}
# parametrs = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 30),
#              'min_samples_split': range(2, 100, 10), 'min_samples_leaf': range(2, 100, 10)}
# grid_search_cv_clf = GridSearchCV(clf, parametrs, cv=5)
# grid_search_cv_clf.fit(X_train, y_train)
# # # print(grid_search_cv_clf.best_params_)
#
# best_clf = grid_search_cv_clf.best_estimator_
# # print(best_clf.score(X_test, y_test))
# y_pred = best_clf.predict(X_test)
# print(precision_score(y_test, y_pred))
# y_predicted_prob = best_clf.predict_proba(X_test)
# y_pred = np.where(y_predicted_prob[:, :] > 0.8, 1, 0)

# from sklearn.metrics import roc_curve, auc
# fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

param_rf = {'n_estimators': range(5, 31, 2), 'max_depth': range(1, 20, 2),
            'min_samples_split': range(2, 101, 5), 'min_samples_leaf': range(2, 101, 5)}
# param_rf = {'max_depth': [13], 'min_samples_leaf': [2], 'min_samples_split': [2], 'n_estimators': [20]}
clf_rf = RandomForestClassifier()
grid_search_cv_clf = GridSearchCV(clf_rf, param_rf, cv=5)
grid_search_cv_clf.fit(X_train, y_train)
best_clf = grid_search_cv_clf.best_estimator_

# фичи по важности
# feature_importances = best_clf.feature_importances_
# feature_importances_df = pd.DataFrame({'features': list(X_train), 'feature_importances': feature_importances})
print(grid_search_cv_clf.best_params_)
print(best_clf.score(X_test, y_test))

res = best_clf.predict(x_Test)
res = list(res)

res_data = pd.DataFrame({'PassengerId': pas_id, 'Survived': res})
filename = "gender_submission.csv"
# print(res_data)
res_data.to_csv(filename, sep=',', index=False)








# рещающий вариант
# clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=100, min_samples_leaf=10)
# clf.fit(X_train, y_train)

