# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.


## Program:
```
## Developed by: Thanushree M
## RegisterNumber: 212224240169

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/content/Placement_Data.csv')
dataset

dataset.head()

dataset.info()

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset.info()

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset.info()

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

dataset.info()

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
dataset.head()

X_train.shape
Y_test.shape

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)

y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_pred,Y_test))
print(confusion_matrix(y_pred,Y_test))

clf.predict([[0,87,0,95,0,2,78,2,0,0,1,0]])

```
## Output:
![image](https://github.com/user-attachments/assets/28019a11-9867-4d98-8035-33cc1fa5568a)
![image](https://github.com/user-attachments/assets/c7c44598-2964-4f50-9e9c-5e0cfca95bdc)
![image](https://github.com/user-attachments/assets/fa63d4be-b4d0-4896-9491-102808f2e02e)
![image](https://github.com/user-attachments/assets/194c9e97-125e-4b09-a6eb-cb8dc65353a4)
![image](https://github.com/user-attachments/assets/d7425287-c142-40c1-b58c-c030f048ae48)
![image](https://github.com/user-attachments/assets/04a42d8f-e259-40ab-82b6-013a6bdd055a)
![image](https://github.com/user-attachments/assets/f9a87e63-f722-4479-97d0-a64cf41a186d)
![image](https://github.com/user-attachments/assets/de20e15c-82db-4321-b0c4-cc11b31233c8)
![image](https://github.com/user-attachments/assets/655a7515-7d60-48ec-a11d-1bc5f3228d20)
![image](https://github.com/user-attachments/assets/4da0fef3-f988-488c-ab0a-06021dbec429)
![image](https://github.com/user-attachments/assets/55ba77c2-32cb-4558-b96f-840f5754133f)
![image](https://github.com/user-attachments/assets/21a9f5fe-134a-49fd-b12d-eee6a1da0a7f)
![image](https://github.com/user-attachments/assets/c8843ccb-a9d1-4ba7-9e98-015f34ba2bc6)
![image](https://github.com/user-attachments/assets/32398ae7-e15f-4c84-b41f-317568372f98)
![image](https://github.com/user-attachments/assets/8fd78d37-0a46-4688-87c1-c9202c312f6a)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
