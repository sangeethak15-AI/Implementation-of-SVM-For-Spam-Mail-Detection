# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sangeetha.K
RegisterNumber: 212221230085 
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![q1](https://user-images.githubusercontent.com/93992063/204533287-64be9d9f-104e-4913-8b46-28b202298526.png)


![2](https://user-images.githubusercontent.com/93992063/204023190-5a1135d2-18ba-4597-bf56-07b34b649612.png)

![3](https://user-images.githubusercontent.com/93992063/204023204-c4709570-1cf9-42bf-bdde-a5870fe5ea4d.png)

![4](https://user-images.githubusercontent.com/93992063/204023231-517a6825-a046-4e9f-8cde-49c378cbc918.png)


![q5](https://user-images.githubusercontent.com/93992063/204533402-7fae8e08-ebd2-4919-8d88-268a93e6332a.png)


![q6](https://user-images.githubusercontent.com/93992063/204533429-3d8ef49d-e92a-4698-b234-1ea2ff7ec35b.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
