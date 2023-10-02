# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate
4. Plot the Cost function using Gradient Descent and generate the required graph.


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Deeksha P
RegisterNumber:  212222040031
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1],color="cadetblue")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000) ")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  """
  take in a numpy array X,y theta and generate the cost function of using the
  in a linear regression model
  """
  m=len(y) #length of the training data
  h=x.dot(theta) #hypothesis
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err) #returning ]

data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta) #call the function

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    j_history.append(computeCost(x,y,theta))
  return theta,j_history

theta,j_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1],color="cadetblue")
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value)
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000,we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000,we predict a profit of $" +str(round(predict2,0)))
```

## Output:

1.profit prediction

![image](https://github.com/Deeksha78/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116204/c195b95f-2007-456c-bf08-5cc73b8cdbfb)



2.function output

![image](https://github.com/Deeksha78/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116204/d3bc359e-b638-408b-ba6d-173113bf1942)


3.Gradient Descent

![image](https://github.com/Deeksha78/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116204/8f14babb-3729-4cf7-b2b1-63adca72ca6b)


4.Cost function using gradient descent

![image](https://github.com/Deeksha78/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116204/55e9f485-77da-4002-b744-48e420542fc5)


5.Linear regression using profit prediction

![image](https://github.com/Deeksha78/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116204/ef4fd45c-d79f-4be0-9ca2-d8bcbeb70c01)


6.Profit prediction for a population of 35,000

![image](https://github.com/Deeksha78/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116204/33bc6c3d-2783-4e47-91f0-0fe90488d0ba)


7.Profit prediction for a population of 70,000

![image](https://github.com/Deeksha78/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116204/36c007cb-faca-47e2-881c-9b5eeca78f56)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
