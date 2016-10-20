#######
#
#  Данный файл представляет из себя непосредственно сам классификатор веб страниц.
#  ...И он не работает
#  Точнее говоря он выдает результат, но тот не верен. Главной проблемой,
#  из-за которой мы получаем такой результат, мне видится датасет. Так как он 80 годов,
#  подозреваю, что лексика в нем устарела.
#
#  По моему скромному мнению, решением является использование нового датасета.
#  Возможно бы было найти открытый API какого-нибудь новостного сайта и скачать там несколько тысяч статей
#  из каждой категории. Таким образом мы будем тренировать классификатор на более релевантных данных.
#
#  Правда при первом поиске открытого API мной были найдены либо платные, либо с ограниченем в 1000 запросов, что
#  означает, что составление датасета заняло бы несколько дней. Это все решаемые проблемы, так что,
#  если у меня оставалось время, я занялся бы этим.
#
#######
from urllib import request
import bs4
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import numpy as np
import sys

from misc_functions import prepare_text_for_analysis, transform_classes

###
#
#  Скачивание самой веб страницы и ее парсинг.
#
###
def get_page(URL):
    raw_page = request.urlopen(URL).read()

    soup = bs4.BeautifulSoup(raw_page, 'html.parser')

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()

    return text

###
#
#  Здесь мы непосредственно загоням векторизированную страницу в классификтор и предоставляем
#  результат пользователю.
#
###
if __name__ == "__main__":
    print("Hello there human!\nI am here to classify your webpage\n" +
          "I am so incredibly efficient at it, that you would be incredibly amazed.")
    #URL = sys.argv[1]

    page = get_page("http://www.agrimoney.com/news/grain-futures-tumble-despite-us-stocks-figures-falling-short-of-forecasts--10026.html")

    prepared_text = prepare_text_for_analysis(page)

    vocabulary = joblib.load("classify_cache/vocab")
    table_of_classes = joblib.load("classify_cache/table_of_classes")
    classifier = joblib.load("classify_cache/classifier")

    vectorizer = TfidfVectorizer(min_df=3, use_idf=True, smooth_idf=False, vocabulary=vocabulary)

    vectorized_text = vectorizer.fit_transform((prepared_text, ))

    prediction = classifier.predict(vectorized_text)

    raw_classes = np.where(prediction == 1)[1]

    print("And...")
    if len(raw_classes) == 0:
        print("I've failed... Give me another one.")
    else:
        classes = transform_classes(raw_classes, table_of_classes)
        print("Well that was easy. Themes are: " + str(classes))


