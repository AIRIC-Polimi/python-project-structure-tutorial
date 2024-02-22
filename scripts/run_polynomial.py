import argparse
import time
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from my_project.data.preprocess import (k_fold_cross_validation_sets,
                                        train_test_split)
from my_project.metrics import mean_squared_error
from my_project.models import PolynomialRegression, PolynomialRidgeRegression


def parse_config(print_config=False):
    parser = argparse.ArgumentParser(
        prog="run_polynomial.py",
        description="Implementation of a Linear Regression with polynomial features to perform a regression task over a given dataset.",
    )

    parser.add_argument("dataset_path", type=str, help="The path to the dataset to use for the regression.")
    parser.add_argument("--x-name", type=str, default="time", help="The name of the x column in the input dataset.")
    parser.add_argument("--y-name", type=str, default="temp", help="The name of the y column in the input dataset.")
    parser.add_argument(
        "--test-perc", type=float, default=0.2, help="The percentage of data to reserve for the test split."
    )
    parser.add_argument("--poly-degree", type=int, default=15, help="The maximum degree of polynomial to use.")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="The learning rate to use for gradient descent."
    )
    parser.add_argument(
        "--iterations", type=int, default=10000, help="The number of iterations to run the gradient descent."
    )
    parser.add_argument(
        "-k", "--k-fold", type=int, default=10, help="The k to use for k fold crossvalidation of hyperparameters."
    )
    parser.add_argument(
        "--ridge-regression", default=False, action="store_true", help="Whether to use ridge regression regularization."
    )
    parser.add_argument(
        "--min-regularization-factor",
        type=float,
        default=0,
        help="The minimum regularization factor to try when using ridge regression.",
    )
    parser.add_argument(
        "--max-regularization-factor",
        type=float,
        default=0.1,
        help="The maximum regularization factor to try when using ridge regression.",
    )
    parser.add_argument(
        "--regularization-factor-step",
        type=float,
        default=0.01,
        help="The step to use when trying diffent regularization factor values.",
    )
    parser.add_argument(
        "--figures-folder",
        type=str,
        default=path.abspath(path.join(path.dirname(__file__), "../figures")),
        help="The path of the folder where to store figures",
    )

    args = parser.parse_args()

    if print_config:
        max_key_len = max([len(k) for k in args.__dict__])
        print("--- CONFIGURATION ---" + "-" * max_key_len)
        for k, v in args.__dict__.items():
            print(f"{k.replace('_', ' '):<{max_key_len+1}}: {v}")
        print("---------------------" + "-" * max_key_len)

    return args


def run(config):
    # Load temperature data
    data = pd.read_csv(config.dataset_path, sep="\t")

    X = np.atleast_2d(data[config.x_name].values).T  # fraction of the year [0, 1]
    y = data[config.y_name].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_perc)

    if config.ridge_regression:
        # Finding regularization constant using cross validation
        lowest_error = float("inf")
        best_reg_factor = None
        print("Finding regularization constant using cross validation:")
        for reg_factor in np.arange(
            config.min_regularization_factor, config.max_regularization_factor, config.regularization_factor_step
        ):
            cross_validation_sets = k_fold_cross_validation_sets(X_train, y_train, k=config.k_fold)
            mse = 0
            for _X_train, _X_test, _y_train, _y_test in cross_validation_sets:
                model = PolynomialRidgeRegression(
                    degree=config.poly_degree,
                    reg_factor=reg_factor,
                    learning_rate=config.learning_rate,
                    n_iterations=config.iterations,
                )
                model.fit(_X_train, _y_train)
                y_pred = model.predict(_X_test)
                _mse = mean_squared_error(_y_test, y_pred)
                mse += _mse
            mse /= config.k_fold

            # Print the mean squared error
            print("\tMean Squared Error: %s (regularization: %s)" % (mse, reg_factor))

            # Save reg. constant that gave lowest error
            if mse < lowest_error:
                best_reg_factor = reg_factor
                lowest_error = mse

        model = PolynomialRidgeRegression(
            degree=config.poly_degree,
            reg_factor=best_reg_factor,
            learning_rate=config.learning_rate,
            n_iterations=config.iterations,
        )
    else:
        model = PolynomialRegression(
            degree=config.poly_degree, learning_rate=config.learning_rate, n_iterations=config.iterations
        )

    # Make final prediction
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    if config.ridge_regression:
        print("Mean squared error: %s (given by reg. factor: %s)" % (mse, best_reg_factor))
    else:
        print("Mean squared error: %s" % mse)

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap("viridis")

    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.suptitle("Polynomial Ridge Regression" if config.ridge_regression else "Polynomial Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel("Day")
    plt.ylabel("Temperature in Celcius")
    plt.legend((m1, m2), ("Training data", "Test data"), loc="lower right")
    plt.savefig(path.join(config.figures_folder, f"{int(time.time())}_mse.png"))


if __name__ == "__main__":
    run(config=parse_config(print_config=True))
