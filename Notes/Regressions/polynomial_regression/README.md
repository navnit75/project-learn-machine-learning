# INTRODUCTION

```
y = b0 + b1x1 + b2x1^2 + b3x1^3 + ...... + bnx1^2
```

- Usually required when data realtion follows a parabolic pattern
- Usually when we talk about epidemic data, and pandemic data.
- QUESTION: Why polynomial LINEAR regression?
- ANSWER : Because we are not considering the variables x1, x2, y here.
- We are considering a, the relations of coefficient. Which are linear form.

# CODE

#### PREPROCESSING

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/content/drive/MyDrive/Dataset/polynomial_linear_regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
```

#### TRAINING THE LINEAR REGRESSOR MODEL

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
```

#### TRAINING THE POLYNOMIAL REGRESSOR MODEL ON WHOLE DATASET

- Algo :
  - We need to create a matrix which contains data in form of x1,x1^2,x1^3 etc.
  - Implement the linear regressor model on the above data

#### CREATION OF MATRIX

```python
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
```

#### IMPLEMENTING Linear Regression ON IT

```python
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)
```

#### VISUALISING THE LINEAR REGRESSION RESULTS

```python
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```

#### VISUALISING THE POLYNOMIAL REGRESSION RESULTS

```python
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```

#### PREDICTING THE NEW RESULT WITH LINEAR REGRESSION

```
lin_reg.predict([[6.5]])
```

#### PREDICTING THE NEW RESULT WITH POLYNOMIAL REGRESSION

```
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
```

# IMAGES

![image]()
