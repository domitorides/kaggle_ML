import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# функция для обработки обзоров в значимые слова
def review_to_words(raw_review):

    # превращаем в html и получаем текс
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text()

    # оставляем с помощью регулярки только буквы, не буквы заменяем пробелом
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # сплитим по нижнему регистру только слова
    words = letters_only.lower().split()

    # стопслова в сет для быстроты программы
    stops = set(stopwords.words("english"))

    # если слов нет в стоп словах, тогда записываем
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)


train = pd.read_csv('labeledTrainData.tsv', quoting=3, delimiter='\t', header=0)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)


num_reviews = train["review"].size

clean_train_reviews = []

print("Получение чистых строк из тренировочного датасэта...\n")
for i in range(0, num_reviews):
    clean_train_reviews.append(review_to_words(train["review"][i]))
    # print(f'{i + 1}/{num_reviews}')


print("Cоздаём наш мешок слов с помощью CountVectorizer...\n")

vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000)

num_reviews = len(test["review"])
clean_test_reviews = []

print("Получение чистых строк из тестового датасэта...\n")
for i in range(0, num_reviews):
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)




# для изучения наших признаков
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# переконвертируем в маасив для удобства
train_data_features = train_data_features.toarray()

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()


X_train, X_test, y_train, y_test = train_test_split(train_data_features, train["sentiment"], test_size=0.33)
# обучим нашу модель

# for max_depth in range(90, 200, 10):
#     param_rf = {'n_estimators': [max_depth]}
#     clf_rf = RandomForestClassifier()
#     grid_search_cv_clf = GridSearchCV(clf_rf, param_rf, cv=5, verbose=1, n_jobs=-1)
#     grid_search_cv_clf.fit(X_train, y_train)
#     best_clf = grid_search_cv_clf.best_estimator_
#     print(f'{best_clf.score(X_test, y_test)}   {best_clf}')


# выберем наилучший скор и запишем в файл
param_rf = {'n_estimators': [170]}
clf_rf = RandomForestClassifier()
grid_search_cv_clf = GridSearchCV(clf_rf, param_rf, cv=5, verbose=1, n_jobs=-1)
grid_search_cv_clf.fit(X_train, y_train)
best_clf = grid_search_cv_clf.best_estimator_
    # print(f'{best_clf.score(X_test, y_test)}   {best_clf}')

result = best_clf.predict(test_data_features)


output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

output.to_csv("sample_submission.csv", index=False, quoting=3)