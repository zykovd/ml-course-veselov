import time
import pathlib
from typing import Tuple, Callable

import pandas as pd
import seaborn as sns
import sympy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum, unique

from utils.logger import create_logger

LOG_DIR = 'output'
logger = create_logger(name='lab2', logging_mode='DEBUG', file_logging_mode='DEBUG', log_to_file=False,
                       log_location=pathlib.Path(__file__).parent.joinpath(LOG_DIR).absolute())


@unique
class ModeEnum(Enum):
    BASIC = 'basic'
    COMPETITION = 'competition'


class Classification:
    def __init__(self, x: np.ndarray, y: np.ndarray, plot_data_2d: bool = True, plot_data_3d: bool = True,
                 plot_correlation: bool = True, num_features: int = 3) -> None:
        self.x = x
        self.y = y

        self.unique_y = np.unique(self.y)

        if plot_correlation:
            self._plot_features_correlation()
            self._plot_features_labels_correlation()

        if plot_data_2d:
            self._plot_pca_visualization(n_components=2)
            self._k_best_features(num_features=2)

        if plot_data_3d:
            self._plot_pca_visualization(n_components=3)
            self._k_best_features(num_features=3)

        self._k_best_transform(num_features=num_features)

    def _k_best_transform(self, num_features: int = 3):
        selector = SelectKBest(score_func=mutual_info_classif, k=num_features)
        selector.fit(X=self.x, y=self.y)
        self.features = selector.get_support(indices=True)
        logger.info("Chosen columns: {}".format(self.features))
        self.x = selector.transform(X=self.x)

    def transform(self, x: np.ndarray):
        return x[:, self.features]

    @staticmethod
    def add_features(x: np.ndarray, num_features: int, add_sin: bool = False, add_log: bool = False) -> np.ndarray:
        """
        num_features = 2 --> Θx^2 + Θx^1 + Θx^0
        ...
        num_features = n --> Θx^n + ... + Θx^1 + Θx^0
        :param x:
        :param num_features:
        :param add_sin:
        :param add_log:
        :return:
        """
        if num_features < 1:
            return x
        #     raise ValueError("'num_features' must be > 1")

        if len(x.shape) == 1:
            temp = np.vstack([x ** deg for deg in range(num_features, -1, -1)])
            if add_sin:
                temp = np.vstack((np.sin(x), temp))
            if add_log:
                temp = np.vstack((np.log2(x), temp))
            return temp.T
        else:
            if num_features == 1:
                return np.hstack([x, np.ones([x.shape[0], 1])])
            else:
                logger.warning("Adding num_features more than {}!".format(num_features))
                temp = np.hstack([x ** deg for deg in range(num_features, 0, -1)])
                temp = np.hstack((temp, np.ones((temp.shape[0], 1))))
                if add_sin:
                    temp = np.hstack((np.sin(x), temp))
                if add_log:
                    temp = np.hstack((np.log2(x), temp))
                return temp

    @staticmethod
    def standardize(train_data: np.ndarray, test_data: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)
        logger.debug("mean = {}".format(mean))
        logger.debug("std = {}".format(std))
        return ((train_data - mean) / std, (test_data - mean) / std) if test_data is not None \
            else ((train_data - mean) / std, None)

    @staticmethod
    def sigmoid(x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def net_input(x: np.ndarray, theta: np.ndarray):
        return np.dot(x, theta)

    @staticmethod
    def probability(x: np.ndarray, theta: np.ndarray):
        return Classification.sigmoid(x=Classification.net_input(x=x, theta=theta))

    @staticmethod
    def cost_function(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(y * np.log(Classification.probability(x=x, theta=theta)) +
                                       (1 - y) * np.log(1 - Classification.probability(x=x, theta=theta)))
        return total_cost

    @staticmethod
    def gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray):
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, Classification.sigmoid(x=Classification.net_input(x=x, theta=theta)) - y)

    @staticmethod
    def predict(x: np.ndarray, theta: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = Classification.probability(x=x, theta=theta)
        classes = (probs >= threshold).astype(int)
        return classes

    def gradient_descent(self, gradient_func, learning_rate: float = 0.00005, epochs: int = 250,
                         plot_error: bool = True):

        t0 = time.time()
        err_all = []
        degree = self.x.shape[1] - 1
        theta = np.zeros((degree + 1))
        for progress in np.arange(0, epochs):
            gradient = gradient_func(x=self.x, y=self.y, theta=theta)
            err_all.append(self.cost_function(x=self.x, y=self.y, theta=theta))
            theta -= learning_rate * gradient
            if progress % 100 == 0:
                print("\rGradient descent progress: {:.1f}% Error: {:.5f} Epoch: {}"
                      .format(progress / epochs * 100, err_all[-1], progress), end="")
        tf = time.time()
        print("\r")
        logger.info('Gradient descent took {} s'.format(tf - t0))
        logger.info('theta = {}'.format(theta))

        if plot_error:
            plt.figure(figsize=(12, 6))
            plt.plot(err_all)
            plt.xlabel('Iteration #')
            plt.ylabel('Error')
            plt.title('Error')
            plt.legend(['Error'])
            plt.grid()

        return theta

    def _plot_pca_visualization(self, n_components: int = 3):
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(self.x)
        fig_title = "PCA {}D".format(n_components)
        figure_pca = plt.figure(fig_title, figsize=(12, 12))

        if n_components == 2:
            y_dataframe = pd.DataFrame(data=self.y, columns=["Class"])
            pca_dataframe = pd.DataFrame(data=principal_components, columns=["Component1", "Component2"])
            pca_dataframe = pd.concat([pca_dataframe, y_dataframe], axis=1)

            ax = figure_pca.add_subplot(1, 1, 1)
            for cluster in self.unique_y:
                indices = pca_dataframe["Class"] == cluster
                ax.scatter(pca_dataframe.loc[indices, "Component1"],
                           pca_dataframe.loc[indices, "Component2"],
                           label="Class {}".format(int(cluster)))

            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title(fig_title)
            ax.grid()
            ax.legend()

        elif n_components == 3:
            y_dataframe = pd.DataFrame(data=self.y, columns=["Class"])
            pca_dataframe = pd.DataFrame(data=principal_components, columns=["Component1", "Component2", "Component3"])
            pca_dataframe = pd.concat([pca_dataframe, y_dataframe], axis=1)

            ax = Axes3D(figure_pca)
            for cluster in self.unique_y:
                indices = pca_dataframe["Class"] == cluster
                ax.scatter(pca_dataframe.loc[indices, "Component1"],
                           pca_dataframe.loc[indices, "Component2"],
                           pca_dataframe.loc[indices, "Component3"],
                           label="Class {}".format(int(cluster)))

            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            ax.set_title(fig_title)
            ax.grid()
            ax.legend()

        return figure_pca

    def _plot_features_correlation(self):
        data = pd.DataFrame(data=self.x, columns=["Column{}".format(i) for i in range(self.x.shape[1])])
        corr_matrix = data.corr()

        figure_corr = plt.figure("Correlation", figsize=(12, 12))
        sns.heatmap(corr_matrix, annot=True)

        return figure_corr

    def _plot_features_labels_correlation(self):
        data = pd.DataFrame(data=self.x,
                            columns=["Column{}".format(i) for i in range(self.x.shape[1])])
        data["Target"] = self.y

        corr = data.drop("Target", axis=1).apply(lambda x: x.corr(data.Target))
        logger.info("Correlation between features and labels:\n{}".format(corr))

        sns.set_theme(style="whitegrid")
        figure_corr = plt.figure("Correlation with labels", figsize=(12, 12))
        sns.barplot(corr.values, corr.index)

        return figure_corr

    def _k_best_features(self, num_features: int = 3):
        x_new = SelectKBest(score_func=mutual_info_classif, k=num_features).fit_transform(self.x, self.y)
        # x_new = SelectKBest(score_func=f_classif, k=num_features).fit_transform(self.x, self.y)
        fig_title = "K-best {}D".format(num_features)
        figure_k_best = plt.figure(fig_title, figsize=(12, 12))

        if num_features == 2:
            y_dataframe = pd.DataFrame(data=self.y, columns=["Class"])
            pca_dataframe = pd.DataFrame(data=x_new, columns=["Component1", "Component2"])
            pca_dataframe = pd.concat([pca_dataframe, y_dataframe], axis=1)

            ax = figure_k_best.add_subplot(1, 1, 1)
            for cluster in self.unique_y:
                indices = pca_dataframe["Class"] == cluster
                ax.scatter(pca_dataframe.loc[indices, "Component1"],
                           pca_dataframe.loc[indices, "Component2"],
                           label="Class {}".format(int(cluster)))

            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title(fig_title)
            ax.grid()
            ax.legend()

        elif num_features == 3:
            y_dataframe = pd.DataFrame(data=self.y, columns=["Class"])
            pca_dataframe = pd.DataFrame(data=x_new, columns=["Component1", "Component2", "Component3"])
            pca_dataframe = pd.concat([pca_dataframe, y_dataframe], axis=1)

            ax = Axes3D(figure_k_best)
            for cluster in self.unique_y:
                indices = pca_dataframe["Class"] == cluster
                ax.scatter(pca_dataframe.loc[indices, "Component1"],
                           pca_dataframe.loc[indices, "Component2"],
                           pca_dataframe.loc[indices, "Component3"],
                           label="Class {}".format(int(cluster)))

            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            ax.set_title(fig_title)
            ax.grid()
            ax.legend()

        return figure_k_best

    @staticmethod
    def accuracy(y_true: np.ndarray, y_hat: np.ndarray) -> float:
        matches = np.sum((y_true == y_hat).astype(int))
        return matches / y_true.shape[0]


def plot_generalization_ability_select(x_train: np.ndarray, y_train: np.ndarray,
                                       x_valid: np.ndarray, y_valid: np.ndarray,
                                       max_features: int = 10):
    """
    Returns generalization ability plot (X: number of features, Y: MSE)
    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :param max_features: Max number of features
    :return:
    """
    list_accuracy_train = []
    list_accuracy_validation = []
    list_error_train = []
    list_error_validation = []
    list_features = range(1, max_features)
    for num_features in list_features:
        classification = Classification(x=x_train, y=y_train, plot_data_2d=False, plot_data_3d=False,
                                        plot_correlation=False, num_features=num_features)

        theta = classification.gradient_descent(gradient_func=Classification.gradient, learning_rate=10e-5,
                                                epochs=10 ** 5, plot_error=False)

        y_hat = Classification.predict(x=classification.transform(x=x_train_np_base), theta=theta)
        accuracy = Classification.accuracy(y_true=y_train, y_hat=y_hat)
        logger.info("accuracy = {} (train)".format(accuracy))
        list_accuracy_train.append(accuracy)
        list_error_train.append(Classification.cost_function(x=classification.x, y=classification.y, theta=theta))

        y_hat = Classification.predict(x=classification.transform(x=x_valid), theta=theta)
        accuracy = Classification.accuracy(y_true=y_valid, y_hat=y_hat)
        logger.info("accuracy = {} (validation)".format(accuracy))
        list_accuracy_validation.append(accuracy)
        list_error_validation.append(Classification.cost_function(x=classification.transform(x=x_valid),
                                                                  y=y_valid, theta=theta))

    figure_ga, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    figure_ga.suptitle('Generalization ability')

    ax1.plot(list_features, list_accuracy_train, 'b', label='Train')
    ax1.plot(list_features, list_accuracy_validation, 'r', label='Validation')
    ax1.set_xlabel('Features (total)')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid()

    ax2.plot(list_features, list_error_train, 'b', label='Train')
    ax2.plot(list_features, list_error_validation, 'r', label='Validation')
    ax2.set_xlabel('Features (total)')
    ax2.set_ylabel('Cost')
    ax2.legend()
    ax2.grid()

    return figure_ga


def plot_generalization_ability(x_train: np.ndarray, y_train: np.ndarray,
                                x_valid: np.ndarray, y_valid: np.ndarray,
                                max_additional_features: int = 3):
    """
    Returns generalization ability plot (X: number of features, Y: MSE)
    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :param max_additional_features: Max number of features
    :return:
    """
    list_accuracy_train = []
    list_accuracy_validation = []
    list_error_train = []
    list_error_validation = []
    list_features = range(1, max_additional_features)
    for num_features in list_features:
        classification = Classification(x=x_train, y=y_train, plot_data_2d=False, plot_data_3d=False,
                                        plot_correlation=False, num_features=3)

        classification.x = Classification.add_features(x=classification.x, num_features=num_features)
        x_valid_ext = classification.transform(x=x_valid)
        x_valid_ext = Classification.add_features(x=x_valid_ext, num_features=num_features)

        theta = classification.gradient_descent(gradient_func=Classification.gradient, learning_rate=10e-5,
                                                epochs=10 ** 5, plot_error=False)

        y_hat = Classification.predict(x=classification.x, theta=theta)
        accuracy = Classification.accuracy(y_true=y_train, y_hat=y_hat)
        logger.info("accuracy = {} (train)".format(accuracy))
        list_accuracy_train.append(accuracy)
        list_error_train.append(Classification.cost_function(x=classification.x, y=classification.y, theta=theta))

        y_hat = Classification.predict(x=x_valid_ext, theta=theta)
        accuracy = Classification.accuracy(y_true=y_valid, y_hat=y_hat)
        logger.info("accuracy = {} (validation)".format(accuracy))
        list_accuracy_validation.append(accuracy)
        list_error_validation.append(Classification.cost_function(x=x_valid_ext, y=y_valid, theta=theta))

    figure_ga, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    figure_ga.suptitle('Generalization ability')

    ax1.plot(list_features, list_accuracy_train, 'b', label='Train')
    ax1.plot(list_features, list_accuracy_validation, 'r', label='Validation')
    ax1.set_xlabel('Features (additional)')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid()

    ax2.plot(list_features, list_error_train, 'b', label='Train')
    ax2.plot(list_features, list_error_validation, 'r', label='Validation')
    ax2.set_xlabel('Features (additional)')
    ax2.set_ylabel('Cost')
    ax2.legend()
    ax2.grid()

    return figure_ga


if __name__ == '__main__':

    try:

        mode = ModeEnum.BASIC
        logger.info("Mode: {}".format(mode.value))

        dump_to_csv = False
        logger.info("dump_to_csv: {}".format(dump_to_csv))

        calc_ga = False
        logger.info("calc_ga: {}".format(calc_ga))

        if mode == ModeEnum.BASIC:
            data_dir = "task2"
            output_dir = "output"
            variant = 3

            test_features_filename = "{}/test_features_{:04d}.csv".format(data_dir, variant)
            test_labels_filename = "{}/lab2.csv".format(output_dir)
            train_features_filename = "{}/train_features_{:04d}.csv".format(data_dir, variant)
            train_labels_filename = "{}/train_labels_{:04d}.csv".format(data_dir, variant)

            logger.info("test_features_filename = {}".format(test_features_filename))
            logger.info("test_labels_filename = {}".format(test_labels_filename))
            logger.info("train_features_filename = {}".format(train_features_filename))
            logger.info("train_labels_filename = {}".format(train_labels_filename))

            x_train = pd.read_csv(train_features_filename, header=None)
            y_train = pd.read_csv(train_labels_filename, header=None)
            x_test = pd.read_csv(test_features_filename, header=None)

            x_train_std, x_test_std = Classification.standardize(train_data=x_train.to_numpy(),
                                                                 test_data=x_test.to_numpy())
            # x_train_std, x_test_std = x_train.to_numpy(), x_test.to_numpy()

            x_train_np_base, x_test_np_base = x_train_std, x_test_std

            total_num_points = x_train_np_base.shape[0]
            logger.debug("total_num_points = {}".format(total_num_points))
            train_percent = 80
            train_points = total_num_points * train_percent // 100

            x_validation_np_base = x_train_np_base[train_points:]
            y_validation = y_train[0].to_numpy()[train_points:]
            x_train_np_base = x_train_np_base[:train_points]
            y_train = y_train[0].to_numpy()[:train_points]

            logger.info('Train points = {}'.format(y_train.shape))
            logger.info('Validation points = {}'.format(y_validation.shape))

            classification = Classification(x=x_train_np_base, y=y_train, plot_data_2d=True, plot_data_3d=True,
                                            plot_correlation=True)

            theta = classification.gradient_descent(gradient_func=Classification.gradient, learning_rate=10e-5,
                                                    epochs=10 ** 6, plot_error=True)

            y_hat = classification.predict(x=classification.transform(x=x_train_np_base), theta=theta)
            logger.info("accuracy = {} (train)".format(classification.accuracy(y_true=y_train, y_hat=y_hat)))

            y_hat = classification.predict(x=classification.transform(x=x_validation_np_base), theta=theta)
            logger.info("accuracy = {} (validation)".format(classification.accuracy(y_true=y_validation, y_hat=y_hat)))

            if dump_to_csv:
                y_hat_test = classification.predict(x=classification.transform(x=x_test_np_base), theta=theta)
                np.savetxt(test_labels_filename, y_hat_test.astype(np.float32), delimiter=",")

            if calc_ga:
                figure_ga_select = plot_generalization_ability_select(x_train=x_train_np_base, y_train=y_train,
                                                                      x_valid=x_validation_np_base,
                                                                      y_valid=y_validation,
                                                                      max_features=x_train_np_base.shape[1])
                figure_ga_select.savefig("{}/ga_select.png".format(output_dir))

                figure_ga = plot_generalization_ability(x_train=x_train_np_base, y_train=y_train,
                                                        x_valid=x_validation_np_base, y_valid=y_validation,
                                                        max_additional_features=4)
                figure_ga.savefig("{}/ga.png".format(output_dir))

            plt.show()

        elif mode == ModeEnum.COMPETITION:
            raise NotImplementedError("Competition part is not implemented")

    except Exception as e:
        logger.error(e)
