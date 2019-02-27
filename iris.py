import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Carrega dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# Informações sobre o dataset

# quantas entradas e quantos atributos
# print(dataset.shape)

# 20 primeiras entradas
# print(dataset.head(20))

# algumas estatísticas sobre o dataset
print(dataset.describe())

# distribuição de classes
# print(dataset.groupby("class").size())


# Gráficos para entender melhor as entradas no dataset

# gráficos de cada variável individualmente

# dataset.plot(kind = "box", subplots = True, layout = (2,2), sharex = False, sharey = False)
# plt.show()

# histograma
# dataset.hist()
# plt.show()

# scatterplot
# mostra a relação entre as entradas de acordo com 2 variáveis
# pode-se gerar uma matriz de gráficos comparando todas as variáveis 2 a 2
# scatter_matrix(dataset)
# plt.show()


# Algoritmos para criar modelos sobre os dados
# e também estimar a precisão para dados não vistos

# separa os dados em duas partes, 80% serão usados pra teste e 20% pra validação
array = dataset.values
x = array[:, 0:4]
y = array[:, 4]
validation_size = 0.20
seed = 7
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size = validation_size, random_state = seed)

# prepara os testes, será usado o 10-fold cross validation para estimar precisão
# esse teste divide os dados em 10 partes, 9 pra treino e 1 pra teste
# repete isso para todas as combinações de divisão treino-teste

scoring = "accuracy"
# accuracy = corretos / total

# constrói os modelos para avaliar os algoritmos
models = []

# modelos lineares
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))

# modelos não lineares
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# avalia cada modelo
results, names = [], []
for name, model in models:
	# usar a mesma seed pra cada teste faz com que os resultados sejam comparáveis
	# mesma seed = mesma divisão do dataset 
	seed = 7
	kfold = model_selection.KFold(n_splits = 10, random_state = seed)
	cv_results = model_selection.cross_val_score(model, x_train, y_train, cv = kfold, scoring = scoring)
	results.append(cv_results)
	names.append(name)
	print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))	

# compara cada algoritmo
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Escolha do algoritmo e previsões usando o dataset de validação
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_validation)
# print(predictions)
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))