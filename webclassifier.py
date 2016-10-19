from urllib import request
import bs4
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import numpy as np

from misc_functions import prepare_text_for_analysis, transform_classes


def get_page(URL):
    raw_page = request.urlopen(URL).read()

    soup = bs4.BeautifulSoup(raw_page, 'html.parser')

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()

    return text


if __name__ == "__main__":
    page = get_page("https://en.wikipedia.org/wiki/Money_supply")

    prepared_text = prepare_text_for_analysis(page)
    print(prepared_text)
    vocabulary = joblib.load("classify_cache/vocab")
    table_of_classes = joblib.load("classify_cache/table_of_classes")
    classifier = joblib.load("classify_cache/classifier")

    vectorizer = TfidfVectorizer(min_df=3, use_idf=True, max_df=0.9, smooth_idf=False, vocabulary=vocabulary)

    vectorized_text = vectorizer.fit_transform(prepared_text)
    print(vectorized_text)
    prediction = classifier.predict(vectorized_text)
    #print(prediction)
    raw_classes = np.where(prediction == 1)[1]
    #print(raw_classes)
    classes = transform_classes(raw_classes, table_of_classes)

    print(classes)