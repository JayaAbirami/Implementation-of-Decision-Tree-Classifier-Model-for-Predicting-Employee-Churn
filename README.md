# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])
```

## Output:
![image](https://github.com/user-attachments/assets/0179761c-3914-4970-976e-057ebc197792)


![image](https://github.com/user-attachments/assets/29bc0242-d22e-4240-ba18-6cf62c705c99)


![image](https://github.com/user-attachments/assets/aec366fa-6b9c-4d42-a729-75bd357dac91)


![image](https://github.com/user-attachments/assets/78309b36-7a1e-409b-9420-2b876f932f1e)

![image](https://github.com/user-attachments/assets/35aadfdb-9d11-40a2-bed8-4bf3cf319219)


![image](https://github.com/user-attachments/assets/f056ae6c-d62f-4dd9-96b0-c935cf9cd265)

![image](https://github.com/user-attachments/assets/61d5bf98-7b35-4219-88a5-a730128c2c23)


![image](https://github.com/user-attachments/assets/bd4a74e9-1321-4b7c-b980-4177e2da85d8)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
