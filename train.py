#########
#
# Данный файл ответсвеннен непосредственно за обучение классификатора.
#
# Используется OneVsRestClassifier, который представляет из себя Multi-Label классификатор.
# Это позволяет нам относить статьи к нескольким классам сразу.
#
# Так же для более успешной классификации используется Tf-Idf векторизация, вместо обычной.
#
#########

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

import time

from dataset import Dataset


###
#
#  Функция векторизует массив документов и кэширует результат. Возращает на выходе векторизованные
#  документы и "словарный запас" векторизатора. В случае считывания значений из кэша "словарный запас" пустой.
#
###
def tf_idf(docs, filename, vocabulary={}):
    if vocabulary == {}:
        vectorizer = TfidfVectorizer(min_df=3, use_idf=True, max_df=0.9, smooth_idf=False)
    else:
        vectorizer = TfidfVectorizer(min_df=3, use_idf=True, max_df=0.9, smooth_idf=False, vocabulary=vocabulary)

    if Path("training_cache/" + filename).is_file():
        vectorized_values = joblib.load("training_cache/" + filename)
        vocab = {}
    else:
        vectorized_values = vectorizer.fit_transform(docs)
        joblib.dump(vectorized_values, "training_cache/" + filename, compress=9)
        vocab = vectorizer.vocabulary_

    return vectorized_values, vocab

##
#
#  Непосредсвенное получение данных и обучение модели.
#  Кэширование "словарного запаса" и классификтора для использовния непосредтсвенно в вебКлассификаторе.
#
##
if __name__ == '__main__':
    train_docs = []
    train_classes = []

    test_docs = []
    test_classes = []

    print("--- Reading data ---")

    start = time.time()
    dataset = Dataset(min_eic=100)

    train_docs, train_classes = dataset.get_training_data()
    test_docs, test_classes = dataset.get_test_data()
    print("Time of dataset preaparation: " + str(time.time() - start))

    print("--- Preparing data ---")

    train_X, vocab = tf_idf(train_docs, "train_vector")
    train_Y = MultiLabelBinarizer().fit_transform(train_classes)

    test_X, vocab = tf_idf(test_docs, "test_vector", vocab)
    test_Y = MultiLabelBinarizer().fit_transform(test_classes)

    if vocab != {}:
        joblib.dump(vocab, "classify_cache/vocab", compress=9)

    print("--- Training ---")

    classifier = OneVsRestClassifier(LinearSVC())
    classifier.fit(train_X, train_Y)
    joblib.dump(classifier, "classify_cache/classifier", compress=9)

    print("--- Testing ---")

    print("Accuracy: " + str(classifier.score(test_X, test_Y)))
