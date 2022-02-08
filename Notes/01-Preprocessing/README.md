# LIBRARY IMPORT

- PANDAS library needs to be implemented

```python

import pandas as pd
```

- MATPLOT lib needed for plotting graphs

```python
import matplotlib.pyplot as plt
```

- NUMPY libray needs to be implemented

```python
import numpy as np
```

# IMPORTING THE DATASET

- Read the csv file in dataset variable

```python
dataset = pd.read_csv('Data.csv')
```

- Separating the independent variable and dependent variable

```python
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
```

# TAKING CARE OF MISSING DATA

- Basically replacing the missing values with the mean of the values
- Using sklearn 

```python
from sklearn.impute import SimpleImputer
```

- creation of the SimpleImputer object and intialise the following properties as mentioned
- Here we are basically replacing the np.nan values as mean of the data in the column.

```python
  imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean')
```

- Only select the column with integer values

```python
imputer.fit(X[:,1:3])
```

- Assigning back the data. 

```python
X[:,1:3]=imputer.transform(X[:,1:3])
```

---

# ENCODING THE CATEGORICAL DATA:

- Its hard for ML models to understand character based categorical data
- Hence the requirement arises to change these data to the numerical format.
- The process is called as ENCODING of the data.
- For ex. Below `France` is changes to `(1 0 0)`

## VARIETY ONE

---

#### Examplary Data

```
[['France' 44.0 72000.0]
 ['Spain' 27.0 48000.0]
 ['Germany' 30.0 54000.0]
 ['Spain' 38.0 61000.0]
 ['Germany' 40.0 nan]
 ['France' 35.0 58000.0]
 ['Spain' nan 52000.0]
 ['France' 48.0 79000.0]
 ['Germany' 50.0 83000.0]
 ['France' 37.0 67000.0]]
```

- CODE

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
```

- OUTPUT obtained

```
[[1.0 0.0 0.0 44.0 72000.0]
 [0.0 0.0 1.0 27.0 48000.0]
 [0.0 1.0 0.0 30.0 54000.0]
 [0.0 0.0 1.0 38.0 61000.0]
 [0.0 1.0 0.0 40.0 63777.77777777778]
 [1.0 0.0 0.0 35.0 58000.0]
 [0.0 0.0 1.0 38.77777777777778 52000.0]
 [1.0 0.0 0.0 48.0 79000.0]
 [0.0 1.0 0.0 50.0 83000.0]
 [1.0 0.0 0.0 37.0 67000.0]]
```

## VARIETY TWO

---

- Data:

```
['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
```

- Code:
  - If data is in binary, use of LabelEncoder is easy. 

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y =le.fit_transform(y)
print(y)
```

- Output:

```
[0 1 0 0 1 1 0 1 0 1]
```

# TEST AND TRAIN SPLIT

- Use of sklearn again

```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)
```

- @test_size = 20% , train_size = 80%
- @random_state = 1 # Basically denotes the details needs to be randomized

# FEATURE SCALING

- Basically refers to scaling down the data into a range so that , one attribute doesn't dominate
  other attributes.
- **Question arises : Feature scaling should be applied before the split or after the split?**
- **Ans**: After the split
- **Reason**: If we apply the feature scaling before the data, the values scaled will be on the basis of
  the whole data set. But we should note that the data should be scaled train and test separately, to know
  the independent results accurately.
- Meanwhile feature scaling after the test and train split also avoids `INFORMATION LEAKAGE`.

- Two Varieties:
  1.  Standardization : Will put all the values between (-3,+3) --> Works all the times
  2.  Normalization: Will put all the values between (0,1) --> When features have normal distribution

- Method to apply
  1.  Apply the standardization in train dataset
  2.  Whatever the mean and deviation obtained from train dataset
  3.  Transform the testing dataset using same deviation and mean

- Feature Scaling shouldn't be applied to the `dummy variables` **(variables introduced due to categorical data)**
- Reason being , they are already between -3 and +3, and if feature scaling changes the values of these data, there identity would be lost.
- As in there combination represents information, which was in string format.
- Modification of `dummy variables` will lose the representation of `INFORMATION` they wish to convey . What is their use?

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.fit_transform(X_test[:,3:])
```
