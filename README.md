# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Packages and read the .csv file.
2. Define a function named ComputeCost and compute the output.
3. Define a function named gradientDescent and iterate the loop.
4. Predict the required graphs using scatterplots.

## Program:
Program to implement the linear regression using gradient descent.

Developed by:Sreevarsha.D

RegisterNumber:212221040159
```

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function of using theta as parameter
   in a linear regression model
  """
  m=len(y) #length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2
  return 1/(2*m) * np.sum(square_err) #returning 
  
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) #Call the function

def gradientDescent(X,y,theta,alpha,num_iters):
  """
  Take in numpy array X,y and theta and update theta by taking num_iters gradient steps 
  with learning rate of alpha
  return theta and the list of the cost of theta during each iteration
  """
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta, J_history
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) = "+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def predict(x,theta):
  """
  Takes in numpy array of x and theta and return the predicted value of y based on theta
  """
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:

Profit prediction graph

![](https://github.com/sreevarshad/Implementation-of-Linear-Regression-Using-Gradient-Descent/edit/main/README.md)

Compute cost value

![](https://github.com/sreevarshad/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/ML%20exp%20202.png)

h(x) value

![](https://github.com/sreevarshad/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/ML%20exp%20203.png)

Cost function using gradient descent graph

![](https://github.com/sreevarshad/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/ML%20exp%20204.png)

Profit prediction graph

![](https://github.com/sreevarshad/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/ML%20exp%20205.png)

Profit for the population 35,000

![](https://github.com/sreevarshad/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/ML%20exp%20206.png)

Profit for the population 70,000

![](https://github.com/sreevarshad/Implementation-of-Linear-Regression-Using-Gradient-Descent/blob/main/ML%20exp%20207.png)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
