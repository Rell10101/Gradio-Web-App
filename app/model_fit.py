from sklearn.datasets import make_regression
import pandas as pd

X, Y = make_regression(300, n_features=5, n_informative=3, noise=25, random_state=181399)



# создание таблички для удобной работы с данными
data = pd.DataFrame({'x1': X[:, 0],
                     'x2': X[:, 1],
                     'x3': X[:, 2],
                     'x4': X[:, 3],
                     'x5': X[:, 4],
                      'y': Y})
