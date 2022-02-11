# INTRODUCTION
- Started with introduction to the dataset in which there are 50 companies and their profit
- R&D spend, administration, statics, profits are given
- Ur job is to create a model to find the most suitable company for investment , their main criteria being profit
- Our model should tell us profit based on their R&D spend, administration, statics, marketing spend

## ASSUMPTIONS
- linearity
- homoscedasticity
- multivariate normality
- independence of errors
- lack of multicollinearity

## DUMMY VARIABLE

- Usually when we have categorical data , then the coloumn needs to be encoded
- When its encoded its divided various coloumn according to the values
- Ex:

  | City       | Var1 | Var2 |
  | ---------- | ---- | ---- |
  | New york   | 1    | 0    |
  | California | 0    | 1    |

- These are called dummy variables, out of these variables we only need New York or California.
- Because wherever its 0 , its automatically understood that its California and vice versa for New York.
- lets denote New York as D1 and california as D2
- D2 = 1 - D1
- D2 can easily be predicted by D1
- For n dummy variables we will use only n-1 dummy variables because their summation results 1 which is a constant and u know we already have a constant in our linear equation.
- This causes two same independent variables in our equation and this phenomenon is called multicollinearity.
- Due to this always one dummy variable needs to be ommitted.
- IF WE HAVE n dummy variables eliminate 1 , and only use n-1 dummy variable.

## STATISTICAL SIGNIFICANCE

- imagine you are tossing coins

```
h0 --> Coin is fair
h1 --> Coin is not fair
#coins probablility
1 (t) 0.5 |
2 (tt) 0.25 |
3 (ttt) 0.12 |
4 (tttt) 0.06 |
--------------------------> alpha =0.05 (Point at which we start believing something is wrong with the coin, you can say 95% h1 is true, 5% h0 is true)
5 (ttttt) 0.03 |
6 (tttttt) 0.01 V Decreasing
```

## BUILDING A MODEL

- 5 methods to build a model :

  - All in
  - Backward Elimination
  - Forward Selection > Step wise regression
  - Bidirectional Elimination
  - Score Comparison

- ALL - IN :

  - If you have the prior knowledge of the domain
  - If there is a company policy which tells to include all the variables
  - Preparing for BACKWARD ELIMINATION

- BACKWARD ELIMINATION:

  1. Select a significance level to stay in the model (e.g. SL = 0.05)
  2. Fit the full model with all possible predictors
  3. Consider the predictor with highest P - value. If P>SL go to step4 , otherwise go to fin
  4. Remove the predictor 
  5. Fit the model without predictor variable , and again to step 3 
  6. Fin: MODEL IS READY

- FORWARD SELECTION:

  - Select a significance level to stay in the model (e.g. SL = 0.05)
  - Fit all the simple regression models y ~ xn. Select the one with lowest P value
  - Keep this variable and fit all possible models with one extra predictors added to the one you already have. <------+
  - Consider the predictor with the lowest P value . If P< SL go to STEP 3 , otherwise go to FIN-----------------------+
  - FIN : Keep the previous model

- BIDIRECTIONAL ELIMINATION:

  - Select a significance level to enter and to stay in the model e.g SENTER = 0.05, SLSTAY = 0.05
  - Perform the next step of Forward Selection ( new variables must have P<SLENTER to enter)<----+
  - Perform ALL steps of BACWARD elimination (old variables must have P<SLSTAY to stay)----------+
  - No new variables can enter and no old variables can exit.
  - FIN : Your model is ready

- ALL POSSIBLE MODELS: (BRUTE FORCE model)
  - Select a criterion for goodness of fit
  - Construct all possible Regression Models. i.e 2^N-1 total combinations (POWER SET FORMULA).
  - Select the one with the best criterion
  - FIN : Your model is ready

# CODE

#### PREPROCESSING

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

#### IMPORTING THE DATASET

```python
dataset = pd.read_csv('/content/drive/MyDrive/Dataset/multiple_linear_regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
```

#### ENCODING THE CATEGORICAL DATA

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
```

#### SPLITTING THE DATA INTO TRAINING SET AND TEST SET

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

#### BUILDING THE MODEL

- THERE IS NO NEED TO FEATURE SCALE IN MULTIPLE LINEAR REGRESSION MODEL AS THE COEFFICIENT ALWAYS
- TAKES CARE OF THE DATA
- WE DO NOT NEED TO TAKE CARE OF DUMMY VARIABLE AS THERE IS A CLASS EXIST FOR

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
```

#### PREDICTING THE RESULTS

```python
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
```

#### Concatenating the y_pred as well as y_test into an array which looks like given below

```python
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
```

```
[[103015.2  103282.38]
 [132582.28 144259.4 ]
 [132447.74 146121.95]
 [ 71976.1   77798.83]
 [178537.48 191050.39]
 [116161.24 105008.31]
 [ 67851.69  81229.06]
 [ 98791.73  97483.56]
 [113969.44 110352.25]
 [167921.07 166187.94]]
```

# IMAGES

#### Multiple Regression Expression

![image1](/reg_eqn?raw=true)

#### Dummy Variables

![image1](link1)

![image1](link1)

![image1](link1)

![image1](link1)
