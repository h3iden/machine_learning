import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

nomes = ['claims', 'payment']
dataset = pd.read_csv('./swedish_insurance.csv', names = nomes)

# dado x, preve y
x = dataset[[nomes[0]]]
y = dataset[nomes[1]]

lr = LinearRegression()
modelo = lr.fit(x, y)

print(modelo.coef_)
print(modelo.intercept_)

a = modelo.coef_[0]
b = modelo.intercept_

fig = plt.figure()
ax1 = fig.add_subplot(111, label = 'ax1')
ax1.set_xlim(left = 0, right = 450)
ax1.set_ylim(bottom = 0, top = 450)
ax1.set_xticks([])
ax1.set_yticks([])

# desenha reta f(x) = b + ax
x1, y1 = [0, 200], [b + 0*a, b + 200*a]
print(x1)
print(y1)
ax1.plot(x1, y1, '--', label = 'h(x)')

ax2 = fig.add_subplot(111, label = 'ax2', frame_on = False)
ax2.set_title('dataset')
ax2.set_xlabel(nomes[0])
ax2.set_ylabel(nomes[1])
ax2.set_xlim(left = 0, right = 450)
ax2.set_ylim(bottom = 0, top = 450)
ax2.plot(dataset[nomes[0]], dataset[nomes[1]], 'x', label = 'entries', c = 'r')

ax1.legend()

plt.show()

print(modelo.predict(np.array([[53]])))