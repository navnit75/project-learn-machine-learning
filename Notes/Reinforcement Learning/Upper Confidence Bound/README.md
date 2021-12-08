# INTRODUCTION

## THE MULTI ARMED BANDIT PROBLEM:

- Deals with mulitple slot machines ( casinos ) , in which you play to maximise your return.
- Factors in the problem:
- Every machine follows a distribution. ( Graph of money spent and money obtained )
- This graph is everything to us
- But more we will try to know distribution about slot machine, more money would be spent on wrong slots.
- We need to find out quickly .
- Factors in to play : Exploration , Exploitation
- We need to exploration --> which is the best one
- Exploitation --> once you find the best, exploit it to maximise return.
- Regret: When we use non optimal method. It is quantized.

## Doubt :

- in example there is 5 machines taken. Why D5 is considered best?

- Ans.
  I was also confused so I did some googling. I think the machines are defined as having some probability distribution which determines how much cash you get when you pull the lever. The machines are a bit weird, as they typically don't return zero dollars. So those are the probability distributions we saw earlier (i.e. a plot of the amount of cash X against probability P(X) to get that amount of cash). Now the aim is to find the machine with the highest average winning when you pull the lever.
  I think these bars with coloured solid lines here are the true average values (plotted on a Y-axis that's not shown), which we don't know and our aim is to find out what they are and once we're pretty sure start betting on that machine. That's why we do some trial runs which also give us an average (the dotted red lines), however, these are unsure, since we only have limited information. The grey boxes result from some statistical calculation that calculates a range of values in between which the true value might lay based on the (few) trials that we have performed. Given this, the algorithm that's explained in the lecture bets on the machine of which the top of the corresponding grey box is highest, as it might have the highest average return value..

# CODE:

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/content/drive/MyDrive/Dataset/upper_confidence_bound/Ads_CTR_Optimisation.csv')
```

## IMPLEMENTING UCB:

```python
import math
N = 10000
d = 10
ads_selected = []
# Creating a list of zero
numbers_of_selection = [0] * d
sums_of_rewards = [0] * d
total_rewards = 0
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
    if (numbers_of_selection[i] > 0):
    average_reward = sums_of_rewards[i] / numbers_of_selection[i]
    delta_i = math.sqrt(3/2 \* math.log(n+1) / numbers_of_selection[i])
    upper_bound = average_reward + delta_i
    else:
    upper_bound = 1e400
    if ( upper_bound > max_upper_bound ):
    max_upper_bound = upper_bound
    ad = i
    ads_selected.append(ad)
    numbers_of_selection[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_rewards += reward
```

## VISUALIZING THE RESULT:

```
plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel('Ads')
plt.ylabel('Number of times each ads was selected')
plt.show()

```
