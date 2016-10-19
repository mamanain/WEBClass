###########
#
#  Класс, ответственный за обращение с данными. При желании
#  датасет можно заменить не меняя API. Это позволяет в будущем добавить большее
#  количество категорий меньшей кровью.
#
#  В конкретном случае используется датасет Reuter, представляющий из себя статьи на тему экономики,
#  поэтому в данном примере классификатор будет классифицирорвать только статьи с экономических сайтов,
#  причем статьи должны поднимать "классические" темы экономики, вроде курса валют и недвижимости,
#  стартапы и технологичские новшества не классифициуются из-за возраста датасета.
#  Дополнительные категории можо добавить при изменении датасета.
#
#  Прим.: В будущем хотелось бы превести происходящие здесь операции на NumPy массивы, для
#  ускорения работы.
#
##########

from nltk.corpus import reuters
from sklearn.externals import joblib
from pathlib import Path

from misc_functions import prepare_text_for_analysis


class Dataset:

    ###
    #
    # Функиця, находящая уникальные классы в списке и составляющая из них таблицу.
    # Число, представляющая из себя класс, так же является индексом массива, под которым
    # находится имя данного класса.
    #
    # Функция так же составляет таблицу, содержащую информацию о количестве документов в каждой
    # ктегории.
    #
    ###
    def make_table_of_classes(self, raw_classes):
        for instance in raw_classes:
            for _class in instance:
                if self.table_of_classes.count(_class) == 0:
                    self.table_of_classes.append(_class)
                    self.num_of_instances.append(0)

                self.num_of_instances[self.table_of_classes.index(_class)] += 1

    ###
    #
    # Функция, переводящая названия классов в датаесете в соответствующие им цифры из таблицы классов.
    #
    # Функция так же убирает из датасета инстансы, классы которых встречались меньше min_eic раз.
    #
    ###
    def transform_classes(self, raw_classes, type):
        classes = []
        for instance in raw_classes:
            current_classes = []
            for _class in instance:
                if self.num_of_instances[self.table_of_classes.index(_class)] >= self.min_eic:
                    current_classes.append(self.table_of_classes.index(_class))

            if len(current_classes) == 0:
                if type == "train":
                    self.train_docs.pop(len(classes))
                elif type == "test":
                    self.test_docs.pop(len(classes))
            else:
                classes.append(current_classes)

        return classes

    ###
    #
    # В конструкоре происходит непосредтвенное чтение данных, кэширование результатов,
    # преобразовние классов.
    #
    ###
    def __init__(self, min_eic=5):

        self.test_classes = []
        self.test_docs = []
        self.train_classes = []
        self.train_docs = []
        self.table_of_classes = []
        self.num_of_instances = []

        # mininimal encounter in classes
        self.min_eic = min_eic

        if Path("training_cache/train_docs").is_file() and Path("training_cache/train_classes").is_file() \
                and Path("training_cache/test_docs").is_file() and Path("training_cache/test_classes").is_file() \
                and Path("classify_cache/table_of_classes").is_file():
            self.train_docs = joblib.load("training_cache/train_docs")
            self.train_classes = joblib.load("training_cache/train_classes")

            self.test_docs = joblib.load("training_cache/test_docs")
            self.test_classes = joblib.load("training_cache/test_classes")

            self.table_of_classes = joblib.load("classify_cache/table_of_classes")
        else:
            raw_test_classes = []
            raw_train_classes = []

            for doc_id in reuters.fileids():
                if doc_id.startswith("train"):
                    self.train_docs.append(prepare_text_for_analysis(reuters.raw(doc_id)))
                    raw_train_classes.append(reuters.categories(doc_id))
                else:
                    self.test_docs.append(prepare_text_for_analysis(reuters.raw(doc_id)))
                    raw_test_classes.append(reuters.categories(doc_id))

            self.make_table_of_classes(raw_train_classes)
            self.train_classes = self.transform_classes(raw_train_classes, "train")
            self.test_classes = self.transform_classes(raw_test_classes, "test")

            joblib.dump(self.train_docs, "training_cache/train_docs", compress=9)
            joblib.dump(self.train_classes, "training_cache/train_classes", compress=9)
            joblib.dump(self.test_docs, "training_cache/test_docs", compress=9)
            joblib.dump(self.test_classes, "training_cache/test_classes", compress=9)
            joblib.dump(self.table_of_classes, "classify_cache/table_of_classes", compress=9)

    ###
    #
    # Непосредственно открытый API класса, предоставляющий доступ к данным.
    #
    ###
    def get_training_data(self):
        return self.train_docs, self.train_classes

    def get_test_data(self):
        return self.test_docs, self.test_classes
