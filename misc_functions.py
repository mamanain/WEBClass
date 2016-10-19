#######
#
#  Полезные функции, нужные всем.
#
######

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.lancaster import LancasterStemmer


##
#
#  Функция, подгатавливающая текст к векторизации.
#  Проводит стеммизацию и фильтрацию знаков препинания и "stop"-вордов (не уверен в переводе).
#
##
def prepare_text_for_analysis(text):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    for word in words:
        if word in stopwords.words('english'):
            words.remove(word)

    stemmer = LancasterStemmer()

    stem_words = []
    for word in words:
        stem_words.append(stemmer.stem(word))

    prepared = " ".join(stem_words)

    return prepared.lower()


##
#
#  Переводит бинарное представление классов, полученное от модели, в названия по таблице.
#
##
def transform_classes(list_of_indexes, table_of_classes):
    answer = []
    for index in list_of_indexes:
        if index >= len(table_of_classes):
            return -1
        answer.append(table_of_classes[index])
    return answer
