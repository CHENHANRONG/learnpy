import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def true_fun(X):
    return np.cos(1.5 * np.pi * X)


n_samples = 30
np.random.seed(0)
X = np.sort(np.random.rand(n_samples))
X = X.reshape(-1,1)
y = true_fun(X) + np.random.randn(n_samples) * 0.1
y = y.reshape(-1,1)

# polynomial_features = PolynomialFeatures(degree=3, include_bias=True)
# X_new = polynomial_features.fit_transform(X=X, y=y)
# linear_regression = LinearRegression()
# pipeline = Pipeline([("polynomial_features", polynomial_features),
#                      ("linear_regression", linear_regression)])
# pipeline.fit(X[:, np.newaxis], y)
#
# # Evaluate the models using crossvalidation
# scores = cross_val_score(pipeline, X[:, np.newaxis], y,
#                          scoring="neg_mean_squared_error", cv=10)


for x_degree in range(3, 20):
    polynomial_features = PolynomialFeatures(degree=x_degree, include_bias=False)
    X_new = polynomial_features.fit_transform(X=X, y=y)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)
    print("degree = %d, score = %d\n" % (x_degree, scores))


print("Stop")
# for degree in range(2)

