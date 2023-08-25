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
RegisterNumber:  212222100016
*/
# implement a simple regression model for predicting the marks scored by the students

import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset)

# assigning hours to X & Scores to Y
X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![200177597-e6ff825e-710a-40ec-842d-50233234b4d3](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/a784dc70-2996-4b0d-a946-41017f61d1ea)


![203804680-9b787e90-79ac-4ddf-a9d8-ec03b8b88ad2](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/590394f9-6516-487f-820c-28d23383f34a)


![203804727-227f8f8c-d13f-4904-9a3e-df48a2f8e84f](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/1a89c742-7a22-452b-8052-ba8bad4e7c62)


![200177609-a5c4987a-11fa-4426-92a8-aced68c0eb61](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/4e10ec37-f106-44af-947e-1ed6a3b67872)


![200177616-98277779-5896-480e-b9ca-702efb43b4de](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/5a9dd4da-7b47-471c-813b-f1c9a0608cb0)


![200177622-f15e4f5e-0163-47f1-80b2-936d0fd1d347](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/65dcd191-c8ca-4479-954d-b706d5bbcbc0)


![200177626-8323e106-b6de-4688-8186-47e015923feb](https://github.com/Jeevithha/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/123623197/011067b1-9579-47d0-afa0-c85e02dda0d9)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
