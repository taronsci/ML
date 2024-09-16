import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/Users/taron.schisas/Desktop/PycharmProjects/ML/Gasprice.txt", header=None)

data.head()
X1 = data.iloc[:, 0:3]
print(X1)

data.describe()

plt.scatter(data[0],data[3])
plt.xlabel("Miles_Traveled (X1)")
plt.ylabel("trvaelTimeinHr (Y)")

plt.scatter(data[1],data[3])
plt.xlabel("Number of deliveries (X1)")
plt.ylabel("trvaelTimeinHr (Y)")

plt.scatter(data[2],data[3])
plt.xlabel("Gas Price (X3)")
plt.ylabel("trvael Time in Hr (Y)")


def computeCost(X, y, theta):
    """
    Take in a numpy array X,y, theta and generate the cost function of using theta as parameter
    in a linear regression model
    """
    m = len(y)
    predictions = X.dot(theta)
    square_err = (predictions - y) ** 2

    return 1 / (2 * m) * np.sum(square_err)

data_n = data.values
m=len(data_n[:,-1])
X = data_n[:,0:3].reshape(m,3)
X=np.append(np.ones((m,1)), X, axis=1)

y=data_n[:,3].reshape(m,1)

theta=np.zeros((4,1))

computeCost(X,y,theta)
#print(data_n)
#print(theta)
#Y= X.transpose()
print("Dimension of X: ", np.shape(X))
print("Dimension of Theta: ", np.shape(theta))
print("Dimension of Y: ",np.shape(y))
#print(X)
#print(y)


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha

    return theta and the list of the cost of theta during each iteration
    """
    m = len(y)
    J_history = []

    for i in range(num_iters):
        predictions = X.dot(theta)
        error = np.dot(X.transpose(), (predictions - y))
        descent = alpha * 2 / m * error
        theta = theta - descent
        J_history.append(computeCost(X, y, theta))

    return theta, J_history

#The Multiple linear regression equation
theta, J_history = gradientDescent(X,y,theta,0.0001,400000)
print("h(x) ="+str(round(theta[0,0],5))+" + "+str(round(theta[1,0],5))+"x1 + "+str(round(theta[2,0],5))+"x2 + "+str(round(theta[3,0],5))+"x3")

#Visualize the cost function
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
print(X)

# Prediction function
def predict(x, theta):
    """
    Takes in numpy array of x and theta and return the predicted value of y based on theta
    """
    predictions = np.dot(theta.transpose(), x)

    return predictions[0]

#Make prediction using optmized theta values
print(theta)
new_x = np.array([110,5,3.54])
new_x = np.append(np.ones(1),new_x)
predict1=predict(new_x,theta)
print("For miles traveled 110, num_deliveries 5, gas price 3.54,  the travel time is "+str(round(predict1,0)),"Hours")

Y_pred = []
l = len(X)
for i in range(0, l):
    temp = predict(np.array(X[i]), theta)
    Y_pred.append(temp)

print(Y_pred)

y_pred = np.array([Y_pred]).T
#print(y, y_pred)
cost = np.sum((y-y_pred)**2)
print("SSE:",cost)

#print(np.shape(y))

#print(np.shape(y_pred))

Y_mean = np.mean(y)
print(Y_mean)
SST = np.sum((Y_mean - y)**2)
SSR = SST - cost
R_Square = (SSR/SST)*100

print('\nSST= ', SST,'\nSSR= ', SSR, '\nSSE= ', cost, '\nR_Square= ', R_Square)