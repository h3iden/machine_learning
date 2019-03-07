import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()
dataset = pd.DataFrame(boston.data, columns = boston.feature_names)
dataset['price'] = boston.target

print(dataset.describe())

# x = dataset.corr()
# plt.subplots(figsize = (20, 20))
# seaborn.heatmap(x, cmap = 'RdYlGn', annot = True)
# plt.show()

X = dataset.drop('price', axis = 1)
y = dataset['price']

x_train, x_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size = 0.20, random_state = 1)

# tamanho dos dados de treinamento e de validacao
# print(x_train.shape, ' ', y_train.shape, ' ', x_validation.shape, ' ', y_validation.shape)

# prepara modelo e ajusta
lm = LinearRegression(copy_X = True, fit_intercept = True, n_jobs = 1, normalize = False)
lm.fit(x_train, y_train)

# preve resultados para os dados de validacao, comparamos depois com y_validation (valor correto)
y_predict = lm.predict(x_validation)

# plota significancia de cada coeficiente
print(lm.coef_)
coefs = pd.DataFrame({'features' : X.columns, 'coef' : lm.coef_})
coefs = coefs.sort_values(by = ['coef'])
coefs.plot(x = 'features', y = 'coef', kind = 'bar', figsize = (15, 10))
plt.show()

# acuracia do modelo pros dados de treinamento e de validacao
train_score = lm.score(x_train, y_train)
validation_score = lm.score(x_validation, y_validation)

# erro (media quadrada e media absoluta)
mean_squared_error_train = metrics.mean_squared_error(y_validation, y_predict)
mean_absolute_error_train = metrics.mean_absolute_error(y_validation, y_predict)

# anexa os resultados no dataframe e mostra
res = pd.concat([x_validation, y_validation], 1)
res['predicted'] = y_predict
res['prediction error'] = res['price'] - res['predicted']
print(res)