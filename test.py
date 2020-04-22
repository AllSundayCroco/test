import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, line_search

def newton(xk, iter=15):
    sequence = [xk]

    return np.array(sequence)
X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
plt.ylim(-1, 1)
plt.xlim(-1, 1)
Z = np.array([[np.sin((x ** 2) / 2 - (y ** 2 ) / 4 + 3) * np.cos(2 * x + 1 - np.exp(y)) for x, y in zip(vx, vy)] for vx, vy in zip(X, Y)])
plt.contour(X, Y, Z, cmap='plasma', levels=np.linspace(np.min(Z), np.max(Z), 15))

sequence = [np.array([-0.3, 0.2])]
for k in range(15):
    dk = -jac(sequence[-1])
    alpha = line_search(fun, jac, sequence[-1], dk)[0]
    sequence.append(sequence[-1] + alpha * dk)
sequence = np.array(sequence)
plt.plot(sequence[:, 0], sequence[:, 1], marker='o', label='sdm')

sequence = [np.array([-0.3, 0.2])]
for k in range(15):
    sequence.append(sequence[-1] + np.linalg.solve(hess(sequence[-1]), -jac(sequence[-1])))
sequence = np.array(sequence)
plt.plot(sequence[:, 0], sequence[:, 1], marker='v', label='newton')

sequence = [np.array([-0.3, 0.2])]
minimize(fun, sequence[0], method='bfgs', jac=jac, callback=lambda xk: sequence.append(xk))
sequence = np.array(sequence)
plt.plot(sequence[:, 0], sequence[:, 1], marker='^', label='bfgs')
plt.legend()