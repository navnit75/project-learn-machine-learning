# INTRODUCTION

- Random forest is version of ensemble learning.
- Ensemble learning : When you take mulitple algorithm or same algorithm multiple times and put them together to make something much more powerful then the original

## STEPS

- Pick at random K data points from the Training Set
- Build the decision tree assosiated to these K data points
- Choose the number Ntrees of trees you want to build and repeat step1 and step2
- For a new data point, make one of your Ntrees predict the value of Y to for the data point in question, and assign the new data point the average across all of the predicted Y values.

## EXAMPLE

- When there are ppl guessing a number of ballons in the pile of ballons
- You do not go and guess
- We stand near the guy who receives the number of ballons , provided by various contestant.
- At last find the average or median --> Statistically.

# CODE

### IMPORT LIBRARIES

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### IMPORT DATASET

```python
dataset = pd.read_csv('/content/drive/MyDrive/Dataset/random_forest_regression/Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
```

### TRAINING MODEL

```python
# n_estimators --> Defines Number of decision trees

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state=0)
regressor.fit(X,y)
```

### PREDICTING THE RESULT

```python
print(regressor.predict([[6.5]]))
```

### VISUALIZING THE RESULT

```python
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```
