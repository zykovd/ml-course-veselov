import pathlib
import re
import traceback
from collections import Counter
from enum import Enum, unique
from time import time

import numpy as np
from sklearn.model_selection import train_test_split

from utils.logger import create_logger

LOG_DIR = 'output'
logger = create_logger(name='lab3', logging_mode='DEBUG', file_logging_mode='DEBUG', log_to_file=False,
                       log_location=pathlib.Path(__file__).parent.joinpath(LOG_DIR).absolute())


@unique
class ModeEnum(Enum):
    BASIC = 'basic'
    COMPETITION = 'competition'


class TextClassification:
    def __init__(self) -> None:
        self.docs = list()
        self.classes = list()
        self.vocab = list()
        self.logprior = dict()
        self.class_vocab = dict()
        self.loglikelihood = dict()

        # vectorizer = CountVectorizer()
        # x_vectorized = vectorizer.fit_transform(x)
        #
        # # logger.info("X = {}".format(x_vectorized.toarray()))
        # logger.info("Features: {}".format(vectorizer.get_feature_names()))

    def count_cls(self, cls):
        cnt = 0
        for idx, _docs in enumerate(self.docs):
            if self.classes[idx] == cls:
                cnt += 1
        return cnt

    def build_global_vocab(self):
        vocab = []
        for doc in self.docs:
            vocab.extend(self.clean_doc(doc))
        return np.unique(vocab)

    def build_class_vocab(self, _cls):
        curr_word_list = []
        for idx, doc in enumerate(self.docs):
            if self.classes[idx] == _cls:
                curr_word_list.extend(self.clean_doc(doc))

        if _cls not in self.class_vocab:
            self.class_vocab[_cls] = curr_word_list
        else:
            self.class_vocab[_cls].append(curr_word_list)

    @staticmethod
    def clean_doc(doc):
        return re.sub(r'[^a-z\d ]', '', doc.lower()).split(' ')

    def fit(self, x, y):
        self.docs = x
        self.classes = y
        num_doc = len(self.docs)
        uniq_cls = np.unique(self.classes)
        self.vocab = self.build_global_vocab()
        vocab_cnt = len(self.vocab)

        t = time()

        for cls in uniq_cls:
            cls_docs_num = self.count_cls(cls)
            self.logprior[cls] = np.log(cls_docs_num / num_doc)
            self.build_class_vocab(cls)
            class_vocab_counter = Counter(self.class_vocab[cls])
            class_vocab_cnt = len(self.class_vocab[cls])

            for word in self.vocab:
                w_cnt = class_vocab_counter[word]
                self.loglikelihood[word, cls] = np.log((w_cnt + 1) / (class_vocab_cnt + vocab_cnt))

        logger.info('Training finished at {} seconds.'.format(time() - t))

    def predict(self, test_docs):
        output = []

        logprior = self.logprior
        vocab = self.vocab
        loglikelihood = self.loglikelihood
        classes = self.classes

        for doc in test_docs:
            uniq_cls = np.unique(classes)
            dict_sum = dict()

            for _cls in uniq_cls:
                dict_sum[_cls] = logprior[_cls]

                for word in self.clean_doc(doc):
                    if word in vocab:
                        try:
                            dict_sum[_cls] += loglikelihood[word, _cls]
                        except Exception:
                            logger.error(traceback.format_exc())
                            logger.error(dict_sum, _cls)

            result = np.argmax(list(dict_sum.values()))
            output.append(uniq_cls[result])

        return output

    @staticmethod
    def accuracy(prediction, test):
        acc = 0
        test_list = list(test)
        for idx, result in enumerate(prediction):
            if result == test_list[idx]:
                acc += 1

        return acc / len(test)


if __name__ == '__main__':

    try:

        mode = ModeEnum.BASIC
        logger.info("Mode: {}".format(mode.value))

        if mode == ModeEnum.BASIC:
            data_dir = "task3"
            output_dir = "output"
            variant = 3

            test_features_filename = "{}/test_features_{:04d}.csv".format(data_dir, variant)
            test_labels_filename = "{}/lab3.csv".format(output_dir)
            train_features_filename = "{}/train_features_{:04d}.csv".format(data_dir, variant)
            train_labels_filename = "{}/train_labels_{:04d}.csv".format(data_dir, variant)

            logger.info("test_features_filename = {}".format(test_features_filename))
            logger.info("test_labels_filename = {}".format(test_labels_filename))
            logger.info("train_features_filename = {}".format(train_features_filename))
            logger.info("train_labels_filename = {}".format(train_labels_filename))

            with open(train_features_filename) as f:
                x_train = [line.rstrip() for line in f]
            with open(test_features_filename) as f:
                x_test = [line.rstrip() for line in f]
            with open(train_labels_filename) as f:
                y_train = [line.rstrip() for line in f]

            # y_train = pd.read_csv(train_labels_filename, header=None)
            # y_train_np = y_train.to_numpy()

            test_size = 0.1
            x_train, x_valid, y_train, y_valid_np = train_test_split(x_train, y_train,
                                                                     test_size=test_size)
            logger.info("Train: {}, Validation: {}, Test: {}"
                        .format(len(x_train), len(x_valid), len(x_test)))

            t_clf = TextClassification()
            t_clf.fit(x=x_train, y=y_train)
            predictions_valid = t_clf.predict(x_valid)
            logger.info('Accuracy: {}'.format(t_clf.accuracy(predictions_valid, y_valid_np)))

            predictions_test = t_clf.predict(x_test)
            with open(test_labels_filename, 'w') as f:
                for prediction in predictions_test:
                    f.write("{}\n".format(prediction))

        elif mode == ModeEnum.COMPETITION:
            raise NotImplementedError("Competition part is not implemented")

    except Exception:
        logger.error(traceback.format_exc())
