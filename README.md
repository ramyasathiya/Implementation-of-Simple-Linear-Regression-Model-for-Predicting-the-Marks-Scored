# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ramya S
RegisterNumber: 212222040130
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
df.head(10)

x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

y_pred = reg.predict(X_test)
y_pred

Y_test

plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_train,reg.predict(X_train),color='yellow')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_squared_error(Y_test,y_pred)
print("MSE : ",mean_squared_error(Y_test,y_pred))
print("MAE : ",mean_absolute_error(Y_test,y_pred))
print("RMSE : ",np.sqrt(mse))
```
## Output:

![image](https://github.com/user-attachments/assets/d2c6e886-7455-4b82-8336-ddcbf3a2cda6)
![image](https://github.com/user-attachments/assets/23d07e30-33c7-49c7-81b5-6c5114121609)
![image](https://github.com/user-attachments/assets/dbc69cc4-9a6d-41c9-bc33-08bc6e30ffa9)
![image](https://github.com/user-attachments/assets/d29e4750-1658-4b14-b5d4-0e5525382a07)
![image](https://github.com/user-attachments/assets/b3d89b5e-c8aa-434d-b3c2-55c4ea7a2c92)
![image](https://github.com/user-attachments/assets/040f3f4c-b3c9-4046-a48f-88e08ba462eb)
![image](https://github.com/user-attachments/assets/2c4ffd91-f7b3-4d59-9197-7842d7036273)
![image](https://github.com/user-attachments/assets/30225d19-608a-4ef4-9f91-97f835f9ea18)









## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
