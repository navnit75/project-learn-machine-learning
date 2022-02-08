# INTRODUCTION

## DEFINATION:

- Any relation which follows a single line on x and y axis.

```
y = b0 + b1.x1
y: Dependent Variable
x1: Independent Variable
b0 : Constant is nothing but y intercept
b1 : Coefficient
```

- Usual conditions we can imagine that

```
^
|
| +
| +
| +
| +
|+
+------------------->
```

- No data points will corresponds to a single line
- in that condition a ordinary least square method arises

```
  +yi
  |
  |
  |
  |
  |
  +yi(cap)
```

- Any point is denoted by two points yi denotes the actual location and yi(cap) denotes the nearest distance from the line

```
SUM((yi(cap) - yi)^2)
```

- Your line found should be as such that , the resulting
- line should have

```
min ( SUM((yi(cap) - yi)^2) ) , for all the points
```

# CODE:

### DATA PREPROCESSING

#### LIBRARY IMPORT

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

#### READ DATA INTO Pandas FRAME

```python
dataset = pd.read_csv('/content/drive/MyDrive/Dataset/simple_linear_regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

#### SPLITTING THE DATA

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

#### TRAINING THE LINEAR REGRESSION MODEL

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
```

#### PREDICTING THE RESULT

```python
y_pred = regressor.predict(X_test)
```

#### VISUALIZING THE RESULT

#### Scatter is used for plotting the Points

```python
plt.scatter(X_train,y_train,color='red')
```

#### Plot is used for the line

```python
plt.plot(X_train,regressor.predict(X_train),color='blue')

plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

#### PREDICTING THE RESULT FOR SINGLE VALUE

```python
y_pred = regressor.predict([[4]])
print(y_pred)
```

#### OUTPUT

```
[64030.39965754]

```

# IMAGES

#### Linear Regression Expression

![image1](/reg_eqn?raw=true)

#### Graph of Simple linear regression

![image1](link1)

#### Ordinary Least Square

![image1](link1)

#### Standardization Normalization

![image1](link1)

#### Assumption of linear regression

![image1](link1)
