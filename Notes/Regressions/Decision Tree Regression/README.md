# INTRODUCTION

## Decision Tree Regression:

- All the scatter plot of data points are split into regions --> Picture in images
- Split will happen -> If splitting adds information to our predefined information
- Or else if there is no new information obtained using predefined information --> We can end the algorithm
- Its not meant for the linear datasets but much more complex dataset. Complex Data means --> Lot of preprocessing may be required
- NO NEED TO APPLY FEATURE SCALING
- We can refer PREPROCESSING code chapter to analyse the preprocessing methodlogies
- Its not very well adapted to 2D dataset

## CART

- Classification trees and Regression trees

# CODE

### IMPORTING THE LIBRARIES

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as
```

### THE DATASET

```python
dataset = pd.read_csv('/content/drive/MyDrive/Dataset/decision_tree_regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
```

### TRAINING THE MODEL

```python
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)
```

### PREDICTING THE NEW RESULT

```python
print(regressor.predict([[6.5]]))
```

### VISUALIZING THE DECISION TREE RESULT

```python
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

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
