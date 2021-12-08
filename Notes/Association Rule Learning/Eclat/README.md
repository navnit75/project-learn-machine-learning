# INTRODUCTION

- It talks about People who bought also bought ( APRIORI )
- Here we only have SUPPORT
- SUPPORT =

```
# user watchlists containing M
------------------------------
# user watchlists
```

- Basically same as the apriori , instead of lift only support is used.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/content/drive/MyDrive/Dataset/apriori/Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

from apyori import apriori

# min_support : basically how much of the assosiation is required
# For example: Here the data set is weekly.
# Now , I require a product which appear 3 times a day.
# It means 7 * 3 weekly
# Support for 21 ==> 21 / 7501 = 0.0027 ~ 0.003

# min_confidence : rule of thumb is we need to BRUTE force it.
# lift : The undestanding comes with experience in doing project.

# min_length, max_length : : Rules when told would be in form of M1 -> M2 .
# i.e one product in on LHS and one product on the RHS
# Here buy one product a , and get at product b 2



rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift =  3, min_length = 2, max_length = 2)

results = list(rules)

# A single row of the
# RelationRecord(
# 0 : items=frozenset({'light cream', 'chicken'}),
# Means: light cream -> chicken



# 1 : support=0.004532728969470737,
# 2: ordered_statistics=[OrderedStatistic(items_base=frozenset({'light cream'}),
# items_add=frozenset({'chicken'}), confidence=0.29059829059829057, lift=4.84395061728395)])
# Means: light cream -> chicken , with the chances of 29%, support = 0.0045,


results

# Representation is very important
# Comes handy in explaining it to some one else


def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]

    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

# n : rows
# columns: Column which you want the  dataframe needs to be sorted


resultsinDataFrame.nlargest(n= 10, columns = 'Support' )
```
