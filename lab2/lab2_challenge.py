import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from utils.logger import create_logger

LOG_DIR = 'output'
logger = create_logger(name='lab2_challenge', logging_mode='DEBUG', file_logging_mode='DEBUG', log_to_file=False,
                       log_location=pathlib.Path(__file__).parent.joinpath(LOG_DIR).absolute())

data_dir = "challenge2"
output_dir = "output"

test_features_filename = "{}/challenge2_x_test.csv".format(data_dir)
test_labels_filename = "{}/lab2_challenge.csv".format(output_dir)
train_features_filename = "{}/challenge2_x_train.csv".format(data_dir)
train_labels_filename = "{}/challenge2_y_train.csv".format(data_dir)

logger.info("test_features_filename = {}".format(test_features_filename))
logger.info("test_labels_filename = {}".format(test_labels_filename))
logger.info("train_features_filename = {}".format(train_features_filename))
logger.info("train_labels_filename = {}".format(train_labels_filename))

x_train = pd.read_csv(train_features_filename, header=None).to_numpy()
y_train = pd.read_csv(train_labels_filename, header=None).to_numpy()
x_test = pd.read_csv(test_features_filename, header=None).to_numpy()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=21)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_valid)
print(y_pred)


def accuracy(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    matches = np.sum((y_true == y_hat).astype(int))
    return matches / y_true.shape[0]


print("Accuracy = {}".format(accuracy(y_true=y_valid, y_hat=y_pred)))

y_hat_test = classifier.predict(x_test)
print(y_hat_test)

with open(test_labels_filename, 'w') as f:
    for y_hat in y_hat_test:
        f.write("{}\n".format(y_hat))
