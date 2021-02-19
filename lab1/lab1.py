import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from enum import Enum, unique


@unique
class ModeEnum(Enum):
    DEMO = 'demo'
    BASIC = 'basic'
    COMPETITION = 'competition'


class PolyRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray, degree: int = 1, plot_data: bool = True) -> None:
        self.x = x
        self.y = y

        self.w = None

        if degree < 1:
            raise ValueError("degree must be > 1")
        # self.degree = degree  # if x.shape[1] == 1 else x.shape[1]

        if len(x.shape) == 1:
            self.x_stack = np.vstack([x ** deg for deg in range(degree, -1, -1)]).T
        else:
            self.x_stack = np.hstack([x, np.ones((x.shape[0], 1))])

        # print(self.x_stack)
        # print(self.x_stack.shape)

        self.x_sorted = np.array(x)
        self.x_sorted.sort()

        if plot_data:
            plt.figure(figsize=(12, 6))
            plt.scatter(self.x, self.y, marker='x', color='b')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Data')
            plt.legend(['Train'])
            plt.grid()

    def gradient_descent(self, learning_rate: float = 0.00005, epochs: int = 250, plot_error: bool = True,
                         plot_substeps: bool = True):
        if self.w is not None:
            pass  # throw warning

        t0 = time.time()
        w_all = []
        err_all = []
        degree = self.x_stack.shape[1] - 1
        self.w = np.zeros((degree + 1))
        for _ in np.arange(0, epochs):
            w_all.append(self.w)
            err = np.dot(self.x_stack, self.w) - self.y
            err_all.append(np.dot(err, err))
            x_transp_x = np.dot(self.x_stack.T, self.x_stack)
            djdw = np.dot(self.w.T, x_transp_x) - np.dot(self.y.T, self.x_stack)
            self.w = self.w - learning_rate * djdw
        tf = time.time()
        print('Gradient descent took {} s'.format(tf - t0))
        print('w = {}'.format(self.w))

        if plot_error:
            plt.figure(figsize=(12, 6))
            plt.plot(err_all)
            plt.xlabel('Iteration #')
            plt.ylabel('RMS Error')
            plt.title('RMS Error')
            plt.legend(['RMS Error'])
            plt.grid()

        if plot_substeps:
            plt.figure(figsize=(15, 20))
            for i in np.arange(0, 8):
                num_fig = i * 30
                y_pred = w_all[num_fig][degree]
                for j in range(degree):
                    y_pred += w_all[num_fig][j] * self.x_sorted ** (degree - j)
                plt.subplot(4, 2, i + 1)
                plt.scatter(self.x, self.y, marker='x', color='b')
                plt.plot(self.x_sorted, y_pred, color='r')
                title_str = '{} iters'.format(num_fig)
                plt.title(title_str)
                plt.legend(['Data', 'Prediction'])
                plt.grid()

    def check_r2_score(self):
        degree = self.x_stack.shape[1] - 1
        y_pred = self.w[degree]
        for j in range(degree):
            y_pred += self.w[j] * self.x ** (degree - j)
        try:
            print(self.y.shape)
            print(y_pred)
            print(type(y_pred))
            r2 = r2_score(y_true=self.y, y_pred=y_pred)
            print('r2_score = {}'.format(r2))
        except ValueError as e:
            print(e)

    def make_prediction(self, x: np.ndarray, plot_prediction: bool = True) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("run 'gradient_descent' before running 'make_prediction'")

        x_sorted = np.array(x)
        x_sorted.sort()

        degree = self.x_stack.shape[1] - 1
        y_pred = self.w[degree]
        for j in range(degree):
            y_pred += self.w[j] * x_sorted ** (degree - j)

        if plot_prediction:
            plt.figure(figsize=(12, 6))
            plt.plot(self.x, self.y, 'bx', x_sorted, y_pred, 'r')
            title_str = 'Predicted function is {}'.format(self.w)
            plt.title(title_str)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend(['Data', 'Prediction'])
            plt.grid()

        return y_pred


if __name__ == '__main__':

    mode = ModeEnum.COMPETITION
    # mode = ModeEnum.BASIC

    if mode == ModeEnum.DEMO:
        X = 4 * np.random.rand(100)
        Y = 7 * X ** 2 + 2 * X + 1 + 1 * np.random.randn(100)
        regression = PolyRegression(x=X, y=Y, degree=2)
        regression.gradient_descent()
        regression.check_r2_score()
        regression.make_prediction(x=X)
        plt.show()

    elif mode == ModeEnum.BASIC:
        data_dir = "task1"
        variant = 3

        test_features_filename = "{}/test_features_{:04d}.csv".format(data_dir, variant)
        train_features_filename = "{}/train_features_{:04d}.csv".format(data_dir, variant)
        train_labels_filename = "{}/train_labels_{:04d}.csv".format(data_dir, variant)

        x_train = pd.read_csv(train_features_filename, header=None)
        y_train = pd.read_csv(train_labels_filename, header=None)
        x_test = pd.read_csv(test_features_filename, header=None)

        # regression = PolyRegression(x=x_train[0].to_numpy(), y=y_train[0].to_numpy(), degree=3, plot_data=False)
        # regression.gradient_descent(learning_rate=10e-8, epochs=100000, plot_substeps=False)

        # print(x_train.to_numpy().shape)

        regression = PolyRegression(x=x_train[0].to_numpy(), y=y_train[0].to_numpy(), degree=2, plot_data=False)
        regression.gradient_descent(learning_rate=10e-6, epochs=100000, plot_substeps=False)
        regression.check_r2_score()
        regression.make_prediction(x=x_test[0].to_numpy())
        plt.show()

    elif mode == ModeEnum.COMPETITION:
        data_dir = "challenge1"

        x_train_filename = "{}/challenge1_x_train.csv".format(data_dir)
        y_train_filename = "{}/challenge1_y_train.csv".format(data_dir)
        x_test_filename = "{}/challenge1_x_test.csv".format(data_dir)

        x_train = pd.read_csv(x_train_filename, header=None)
        y_train = pd.read_csv(y_train_filename, header=None)
        x_test = pd.read_csv(x_test_filename, header=None)

        # print(x_train.to_numpy().shape)

        regression = PolyRegression(x=x_train.to_numpy(), y=y_train[0].to_numpy(), degree=1, plot_data=False)
        regression.gradient_descent(learning_rate=10e-40, epochs=100000, plot_substeps=False)
        regression.check_r2_score()
        plt.show()
