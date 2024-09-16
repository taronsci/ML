import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8.0, 5.0)

X = np.array([34, 108, 64, 88, 99, 51])
Y = np.array([5, 17, 11, 8, 14, 5])

plt.scatter(X, Y)
plt.show()

theta1_list = []
theta0_list = []
cost_list = []

theta1 = 1
theta0 = 1

L = 10**-5
epochs = 800000
Y_pred = theta0 + theta1*X

m = float(len(X))
iteration = []

#performing gradient descent
print('theta0\t, theta1\t, cost', 'Epoch')
for i in range(epochs):
    Y_pred = theta0 + theta1*X
    temp0 = (2/m)*sum(Y_pred-Y)
    temp1 = (2/m)*sum(X*(Y_pred-Y))

    theta0 = theta0 - L*temp0
    theta1 = theta1 - L*temp1

    theta0_list.append(theta0)
    theta1_list.append(theta1)

    Y_pred = theta0 + theta1*X

    cost = np.sum((Y_pred - Y)**2)
    cost_list.append(cost)

    iteration.append(i)
    if i < 10:
        plt.scatter(X,Y)
        print(theta0, '\t', theta1, '\t', cost)
        plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color = 'red')
        plt.show()

Y_pred = theta0 + theta1*X

print('Linear Regression Equation: Y_pred = ', theta0, '+', theta1,'X')

Y_mean = np.mean(Y)
SST = np.sum((Y_mean - Y)**2)
SSR = SST - cost
R_Square = (SSR/SST) * 100
print('\nSST= ', SST,'\nSSR= ', SSR, '\nSSE= ', cost, '\nR_Square= ', R_Square)