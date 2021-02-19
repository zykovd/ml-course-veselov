import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score

is_use_standardization = True
data_dir = "task1"
variant = 0

test_features_filename = "{}/test_features_{:04d}.csv".format(data_dir, variant)
train_features_filename = "{}/train_features_{:04d}.csv".format(data_dir, variant)
train_labels_filename = "{}/train_labels_{:04d}.csv".format(data_dir, variant)

train_x_df = pd.read_csv(train_features_filename, header=None)
train_y_df = pd.read_csv(train_labels_filename, header=None)

test_x_df = pd.read_csv(test_features_filename, header=None)

plt.figure(figsize=(12, 6))
plt.scatter(train_x_df[0], train_y_df[0], label="Training", color="b", marker="x")

if is_use_standardization:
    x_mu = train_x_df[0].mean()
    y_mu = train_y_df[0].mean()
    x_scale = train_x_df[0].std()
    y_scale = train_y_df[0].std()

    x_train_scale = (train_x_df[0] - x_mu) / x_scale
    x_train_scale_np = x_train_scale.to_numpy().reshape(-1, 1)
    y_train_scale = (train_y_df[0] - y_mu) / y_scale

# "squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"
loss = "epsilon_insensitive"
reg = SGDRegressor(loss=loss, max_iter=1e8, alpha=1e-5, tol=1e-10)

if is_use_standardization:
    reg.fit(x_train_scale_np, y_train_scale)
else:
    reg.fit(train_x_df[0].to_numpy().reshape(-1, 1), train_y_df[0])

x_train_reshaped = train_x_df[0].to_numpy().reshape(-1, 1)
if is_use_standardization:
    y_train_predicted = reg.predict((x_train_reshaped - x_mu) / x_scale) * y_scale + y_mu
else:
    y_train_predicted = reg.predict(x_train_reshaped)
print("r2_score = {}".format(r2_score(y_true=train_y_df[0], y_pred=y_train_predicted)))

x_test_reshaped = test_x_df[0].to_numpy().reshape(-1, 1)
if is_use_standardization:
    yhat = reg.predict((x_test_reshaped - x_mu) / x_scale) * y_scale + y_mu
else:
    yhat = reg.predict(x_test_reshaped)
plt.plot(test_x_df[0], yhat, label="Predictions", color="r", marker="o")

plt.title("Dataset \'{}, variant {}\'".format(data_dir, variant))
plt.grid()
plt.legend()
plt.show()
