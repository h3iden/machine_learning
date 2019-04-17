import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

nomes = ['population', 'profit']
dataset = pd.read_csv('./ex1data1.txt', names = nomes)

X = dataset[[nomes[0]]]
y = dataset[nomes[1]]

clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
clf.fit(X, y)

print('predict 3.5 : ', clf.predict(np.array([[3.5]])))
print('predict 7.0 : ', clf.predict(np.array([[7.0]])))

a = clf.coef_[0]
b = clf.intercept_[0]

fig = plt.figure()
ax1 = fig.add_subplot(111, label = 'ax1')
# ax1.set_xlim(left = -5, right = 30)
# ax1.set_ylim(bottom = -5, top = 30)
ax1.set_xticks([])
ax1.set_yticks([])

# desenha reta f(x) = b + ax
x1, y1 = [0, 1000], [b + 0*a, b + 1000*a]
print(x1)
print(y1)
ax1.plot(x1, y1, '--', label = 'h(x)')

ax2 = fig.add_subplot(111, label = 'ax2', frame_on = False)
ax2.set_title('dataset')
ax2.set_xlabel(nomes[0])
ax2.set_ylabel(nomes[1])
ax2.set_xlim(left = -5, right = 30)
ax2.set_ylim(bottom = -5, top = 30)
ax2.plot(dataset[nomes[0]], dataset[nomes[1]], 'x', label = 'entries', c = 'r')

ax1.legend()

plt.show()