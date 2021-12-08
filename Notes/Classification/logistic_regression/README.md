# INTRODUCTION

- When we talk about the classification the idea of regression differs.
- The image Linear Regression vs Logistic Regression provide insight how the problem look like.
- The images describes how linear regression approach is used to provide a probablistic idea about predictions
- Then later we use sigmoid kind of function to attain a more accurate idea of the prediction.
- This sigmoid function results in large number of sigmodial lines which describes our data.
- Logistic regression finds that one line that best describes the data.
- Logistic regression lines help in predicting the probablities. Check images
- [Logistic Regression](https://www.superdatascience.com/blogs/the-ultimate-guide-to-regression-classification)

## DATASET

- Social advertisement dataset
- You are from a CAR company
- You are given data of previous purchases and salary of a person
- You need to predict if a given person with respective feature will buy the car or not

# CODE

### IMPORT LIBRARIES

````python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### IMPORTING THE DATASET
```python
dataset = pd.read_csv('/content/drive/MyDrive/Dataset/logistic_regression/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

### SPLITTING THE DATASET INTO TRAIN and TEST SET
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
````

### FEATURE SCALING

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### TRAINING THE MODEL

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### PREDICTING THE NEW RESULT

```python
# Shows the prediction of whether customer is going to buy or not

y_new_pred = classifier.predict(sc.transform([[30,87000]]))
print(y_new_pred)

# Shows the probablity of whether customer is going to buy or not

y_new_probab = classifier.predict_proba(sc.transform([[30,87000]]))
print(y_new_probab)
```

### PREDICTING THE TEST RESULT

```python
y_pred = classifier.predict(X_test)
print(np.concatenate( (y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
```

### MAKING THE CONFUSION MATRIX

```python
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)
```

```
# 65 -> Correct Prediction for class 0 (True negative)
# 3 -> Incorrect Prediction for class 0 (False Positive)
# 8 -> Incorrect Prediction for class 1 (False Negative)
# 24 -> Correct Prediction for class 1 (True positive)
```

- OUTPUT

```
[[65  3]
 [ 8 24]]
0.89
```

### VISUALIZING THE TRAINING SET RESULT

```python
from matplotlib.colors import ListedColormap
# Code is pretty advanced , we need not understand each and every line
# Creation of grid here , the grid is made dense for the reason of high clearity , and high resolution graph
# If the color mis match happens --> they are the wrong predicted results
# The line which separates the red region and blue region is called prediction boundary
# Prediction boundary for any linear classification model is always a STRAIGHT LINE
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

### VISUALIZING THE TEST SET RETURN

```python
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

```
