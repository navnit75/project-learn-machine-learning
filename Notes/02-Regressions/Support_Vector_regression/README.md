# INTRODUCTION

- Remains same as the polynomial regression model
- A person wants a job in your company
- Problem is he is telling his level is 6.5 and his salary was 160000 per year
- Our dataset represents salary distribution of the his previous company
- We need to train our model with previous company salary with respect to level, and find his salary for level 6.5
- So that we know he is lying or telling the truth
- Comparison of different programming model will be taking place.

## REQUIREMENT

- When Feature Scaling is not needed:

  - On the dummy variables resulting from ONE HOT ENCODING
  - When a dependent variable takes binary value

- We need to apply feature scaling:

  - When dependent variable takes super high values with respect to other features
  - Meanwhile the models which doesn't have any implicit relation
  - For the models which have an implicit relation between Dependent variable and Independent variable

- Requirement of Feature Scaling in the SVR:
  - Reason: There is no explicit equation which balances our dependency on various variables , which creates an issue for us
  - If not done , the feature taking high values will be preferred resulting in information loss

# CODE

### PREPROCESSING

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/content/drive/MyDrive/Dataset/support \_vector_regression/Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values
```

### Y TRANSFORMATION

- Usually independent variables are in the 2d matrix already --> meanwhile dependent variables are in the vector form or 1d array
- We need to change the y vector into -> matrix format

```python
# y.reshape(#rows,#columns)

y = y.reshape(len(y),1)
```

### FEATURE SCALING

- Requirement of separate scalers
- sc_X will contain all the metadata for example, mean, standard deviation related to data provided in X
- if we use the same scaler, the X mean would be applied to perform feature scaling on y
- hence requirement of separate scaler arises

```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
```

### TRAINING MODEL

```python
from sklearn.svm import SVR

# there are variety of kernel based on the requirement

# Gaussian Radial Basis function is used here

regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)
```

### PREDICTING THE RESULT

- Scaling bring us a problem when predicting the result.
- Because we want the result based on unscaled value .
- For example we want the person's salary on level 6.5.
- But our model has been trained on values which have been scaled in the limit -3 , 3
- To solve this problem there are various methods provided

- Steps
  - Transform 6.5 into 2d array -> [[6.5]]
  - Feature scale it using sc_X -> sc_X.transform([[6.5]])
  - Predict the corresponding y value -> regressor.predict(sc_X.transform([[6.5]]))
  - Provide the value to inverse transform to know the salary -> sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

```python
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]]))))
```

### VISUALIZING THE RESULT

```python
# -- Scatter is used for the Points

plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')

# -- Plot is used for the line

plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)),color='blue')

plt.title('Support Vector Regression')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()
```

### HIGHER RESOLUTION RESULT

```python
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

```
