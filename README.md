# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JEEVITHA S
RegisterNumber: 212222100016  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

# splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

# displaying predicted values
y_pred

plt.scatter(x_train,y_train,color='brown')
plt.plot(x_train,regressor.predict(x_train),color='green')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='purple')
plt.plot(x_test,regressor.predict(x_test),color='red')
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE= ",mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
 ##df.head()
![image](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/e6660457-7c4e-46c9-914a-ab287232f32c)

##df.tail()
![image](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/82a6c704-534c-48d3-930c-e6ba4724613d)
##Array value of X
![image](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/d1b0480e-0a5c-4011-b0eb-b80198decce6)
##Array value of Y
![image](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/63b418ad-fa05-45c8-a696-dfaeb71db6c8)
##Values of Y prediction
![image](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/0303b5d2-ae86-4f13-b755-323de3d3ce7e)
##Array values of Y test
![image](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/31b3e922-9142-4d0b-acab-55f880473428)
##Training Set Graph
![image](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/ff8d3c4b-0043-4a0e-8b24-b532c0cd40d0)
##Test Set Graph
![image](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/1021b83d-20fc-4374-bb3e-55c8ad607c66)
##Values of MSE, MAE and RMSE
![image](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/d629eb9b-e832-444d-bc19-969b152cb04c)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
