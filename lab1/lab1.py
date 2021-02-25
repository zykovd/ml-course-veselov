import time
import pathlib
from typing import Tuple

import pandas as pd
import sympy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from enum import Enum, unique

from utils.logger import create_logger

LOG_DIR = 'output'
logger = create_logger(name='lab1', logging_mode='DEBUG', file_logging_mode='DEBUG', log_to_file=True,
                       log_location=pathlib.Path(__file__).parent.joinpath(LOG_DIR).absolute())


@unique
class ModeEnum(Enum):
    BASIC = 'basic'
    COMPETITION = 'competition'


class LinearRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray, plot_data: bool = True) -> None:
        self.x = x
        self.y = y
        self.feature = x[:, -2]

        if plot_data:
            plt.figure(figsize=(12, 6))
            plt.scatter(self.feature, self.y, marker='x', color='b')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Data')
            plt.legend(['Train'])
            plt.grid()

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
            raise ValueError("'num_features' must be > 1")

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
                temp = np.hstack([x ** deg for deg in range(num_features, -1, -1)])
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

    def gradient_descent(self, learning_rate: float = 0.00005, epochs: int = 250, plot_error: bool = True,
                         plot_substeps: bool = True):

        t0 = time.time()
        w_all = []
        err_all = []
        degree = self.x.shape[1] - 1
        theta = np.zeros((degree + 1))
        for progress in np.arange(0, epochs):
            if plot_substeps:
                w_all.append(theta)
            err = np.dot(self.x, theta) - self.y
            err_all.append(np.dot(err, err))
            x_transp_x = np.dot(self.x.T, self.x)
            djdw = np.dot(theta.T, x_transp_x) - np.dot(self.y.T, self.x)
            theta = theta - learning_rate * djdw
            if progress % 100 == 0:
                print("\rGradient descent progress: {:.1f}% Error: {:.5f} Epoch: {}"
                      .format(progress / epochs * 100, err[1], progress), end="")
        tf = time.time()
        print("\r")
        logger.info('Gradient descent took {} s'.format(tf - t0))
        logger.info('theta = {}'.format(theta))

        if plot_error:
            plt.figure(figsize=(12, 6))
            plt.plot(err_all)
            plt.xlabel('Iteration #')
            plt.ylabel('RMS Error')
            plt.title('RMS Error')
            plt.legend(['RMS Error'])
            plt.grid()

        if plot_substeps:
            num_subplots = 9
            list_iters_for_plot = [1, 2, 3, 10, 11, 12, 500, 2000, 4000]
            # list_iters_for_plot = [i * (epochs // num_subplots) for i in range(num_subplots)]

            plt.figure()
            for i, epoch in enumerate(list_iters_for_plot):
                y_pred = np.dot(self.x, w_all[epoch])
                x_sorted, y_sorted, y_pred_sorted = (np.array(t) for t in zip(*sorted(zip(self.feature,
                                                                                          self.y, y_pred),
                                                                                      key=lambda _t: _t[0])))
                plt.subplot(num_subplots // 3, 3, i + 1)
                plt.scatter(x_sorted, y_sorted, marker='x', color='b')
                plt.plot(x_sorted, y_pred_sorted, color='r')
                title_str = 'Iter {}'.format(epoch)
                plt.title(title_str)
                plt.legend(['Data', 'Prediction'])
                plt.grid()

        return theta

    @staticmethod
    def make_prediction(x: np.ndarray, theta: np.ndarray, y: np.ndarray = None,
                        plot_prediction: bool = False) -> np.ndarray:

        y_pred = np.dot(x, theta)

        if plot_prediction:
            feature = x[:, -2]

            if y is not None:
                x_sorted, y_sorted, y_sorted_pred = (np.array(t) for t in zip(*sorted(zip(feature, y, y_pred),
                                                                                      key=lambda _t: _t[0])))
            else:
                x_sorted, y_sorted_pred = (np.array(t) for t in zip(*sorted(zip(feature, y_pred),
                                                                            key=lambda _t: _t[0])))

            plt.figure(figsize=(12, 6))
            if y is not None:
                plt.plot(feature, y, 'bx')
            plt.plot(x_sorted, y_sorted_pred, 'r')
            title_str = 'Predicted function is {}'.format(theta)
            plt.title(title_str)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend(['Data', 'Prediction'])
            plt.grid()

        return y_pred

    @staticmethod
    def plot_scatter(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                     feature: int = -2):
        plt.figure(figsize=(12, 6))
        plt.plot(x_train[:, feature], y_train, 'rx')
        plt.plot(x_test[:, feature], y_test, 'bo')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(['Train', 'Test'])
        plt.grid()

    @staticmethod
    def get_independent_cols_idx(matrix: np.ndarray) -> Tuple[int]:
        mtx, inds = sympy.Matrix(matrix).rref()
        logger.debug("Matrix: {}".format(mtx))
        return inds

    @staticmethod
    def get_corr_matrix(df: pd.DataFrame) -> np.ndarray:
        num_columns = len(df.columns)
        mtx = np.zeros((num_columns, num_columns))
        for i, i_column in enumerate(df):
            for j, j_column in enumerate(df):
                mtx[i][j] = df[i_column].corr(df[j_column])
                if i != j and mtx[i][j] > 0.8:
                    logger.debug("Correlation between {} and {} is high: {}".format(i, j, mtx[i][j]))
        return mtx


if __name__ == '__main__':

    try:

        # mode = ModeEnum.COMPETITION
        mode = ModeEnum.BASIC
        logger.info("Mode: {}".format(mode.value))

        if mode == ModeEnum.BASIC:
            data_dir = "task1"
            output_dir = "output"
            variant = 3

            test_features_filename = "{}/test_features_{:04d}.csv".format(data_dir, variant)
            test_labels_filename = "{}/lab1.csv".format(output_dir)
            train_features_filename = "{}/train_features_{:04d}.csv".format(data_dir, variant)
            train_labels_filename = "{}/train_labels_{:04d}.csv".format(data_dir, variant)

            logger.info("test_features_filename = {}".format(test_features_filename))
            logger.info("test_labels_filename = {}".format(test_labels_filename))
            logger.info("train_features_filename = {}".format(train_features_filename))
            logger.info("train_labels_filename = {}".format(train_labels_filename))

            x_train = pd.read_csv(train_features_filename, header=None)
            y_train = pd.read_csv(train_labels_filename, header=None)
            x_test = pd.read_csv(test_features_filename, header=None)

            x_train_np, x_test_np = x_train[0].to_numpy(), x_test[0].to_numpy()

            num_features = 2
            x_train_np = LinearRegression.add_features(x=x_train_np, num_features=num_features)
            x_test_np = LinearRegression.add_features(x=x_test_np, num_features=num_features)

            regression = LinearRegression(x=x_train_np, y=y_train[0].to_numpy(), plot_data=False)

            theta = regression.gradient_descent(learning_rate=10e-6, epochs=10000, plot_substeps=True)

            y_hat = regression.make_prediction(x=x_train_np, theta=theta, y=regression.y, plot_prediction=True)
            logger.info("r2_score = {}".format(r2_score(y_true=regression.y, y_pred=y_hat)))

            y_hat_test = regression.make_prediction(x=x_test_np, theta=theta, plot_prediction=False)
            LinearRegression.plot_scatter(x_train=x_train_np, y_train=regression.y, x_test=x_test_np, y_test=y_hat_test)
            pd.DataFrame(y_hat_test).to_csv(test_labels_filename, encoding='utf-8', index=False, header=False)

            plt.show()

        elif mode == ModeEnum.COMPETITION:
            data_dir = "challenge1"
            output_dir = "output"

            x_train_filename = "{}/challenge1_x_train.csv".format(data_dir)
            y_train_filename = "{}/challenge1_y_train.csv".format(data_dir)
            x_test_filename = "{}/challenge1_x_test.csv".format(data_dir)
            y_test_filename = "{}/lab1_challenge_91_10e-9_10v6_2features.csv".format(output_dir)

            logger.info("x_train_filename = {}".format(x_train_filename))
            logger.info("y_train_filename = {}".format(y_train_filename))
            logger.info("x_test_filename = {}".format(x_test_filename))
            logger.info("y_test_filename = {}".format(y_test_filename))

            x_train = pd.read_csv(x_train_filename, header=None)
            y_train = pd.read_csv(y_train_filename, header=None)
            x_test = pd.read_csv(x_test_filename, header=None)

            corr_mtx = LinearRegression.get_corr_matrix(df=x_train)
            logger.debug("Corr matrix: {}".format(corr_mtx.tolist()))
            set_columns_idx_del = set()
            for i in range(corr_mtx.shape[0]):
                for j in range(corr_mtx.shape[1]):
                    if i != j and corr_mtx[i][j] > 0.8:
                        set_columns_idx_del.add(min(i, j))

            logger.warning("Dropping columns: {}".format(set_columns_idx_del))
            x_train.drop(x_train.columns[list(set_columns_idx_del)])
            x_test.drop(x_test.columns[list(set_columns_idx_del)])

            x_train_std, x_test_std = LinearRegression.standardize(train_data=x_train.to_numpy(),
                                                                   test_data=x_test.to_numpy())

            total_num_points = x_train_std.shape[0]
            logger.debug("total_num_points = {}".format(total_num_points))
            train_percent = 75
            train_points = total_num_points * train_percent // 100

            # x_validation_std = x_train_std[train_points:]
            x_validation_std = x_train_std
            # x_train_std = x_train_std[:train_points]
            x_train_std = x_train_std

            num_features = 3
            x_train_np = LinearRegression.add_features(x=x_train_std, num_features=num_features)
            x_validation_np = LinearRegression.add_features(x=x_validation_std, num_features=num_features)
            x_test_np = LinearRegression.add_features(x=x_test_std, num_features=num_features)

            # y_validation_np = y_train[0].to_numpy()[train_points:]
            y_validation_np = y_train[0].to_numpy()
            # y_train_np = y_train[0].to_numpy()[:train_points]
            y_train_np = y_train[0].to_numpy()

            regression = LinearRegression(x=x_train_np, y=y_train_np, plot_data=False)

            theta = regression.gradient_descent(learning_rate=10e-10, epochs=10 ** 7, plot_substeps=False)

            y_hat = regression.make_prediction(x=x_validation_np, theta=theta, plot_prediction=False)
            logger.info("r2_score = {}".format(r2_score(y_true=y_validation_np, y_pred=y_hat)))

            y_hat_test = regression.make_prediction(x=x_test_np, theta=theta, plot_prediction=False)
            pd.DataFrame(y_hat_test).to_csv(y_test_filename, encoding='utf-8', index=False, header=False)

            plt.show()

    except Exception as e:
        logger.error(e)
