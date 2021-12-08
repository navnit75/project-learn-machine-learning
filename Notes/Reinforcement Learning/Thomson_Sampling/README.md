# INTRODUCTION

- We start at a single x axis where all the machines expected (mean) probablities are mapped.
- Now instead of checking the confidence like in Upper Confidence Bound, we started with normal distribution around the expected probablities.
- As we keep adding more and more points from sampling, the activity will be used to improve the accuracy of normal distribution.
- And the machine with highest expected return probablity is chosen.

## Difference between UCB and Thompson Sampling

- UCB's disadvantages remains in the delays.
- After every round there is a requirement of answer for that round.
- But for Thompson sampling , it is dynamic.
- For example:
- In click ad scenerio if we have to do UCB it would be computationally high.
- But for similar situation , we can use thomas sampling. And process the data obtained in batches.
- After every 500 or 5000 clicks update the model.
- Hence flexible.

# CODE

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('/content/drive/MyDrive/Dataset/upper_confidence_bound/Ads_CTR_Optimisation.csv')


import random
N = 500
d = 10
ads_selected = []

number_of_rewards_0 =[0] * d
number_of_rewards_1 =[0] * d

total_reward = 0

for n in range(0,N):
    ad = 0
    max_random = 0
    for i in range(0,d):
      random_beta = random.betavariate(number_of_rewards_1[i]+1 , number_of_rewards_0[i] + 1)

      if(random_beta > max_random):
        max_random = random_beta
        ad  = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if( reward == 0):
      number_of_rewards_0[ad] += 1
    else:
      number_of_rewards_1[ad] += 1
    total_reward += reward


plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel('Ads')
plt.ylabel('Number of times each ads was selected')
plt.show()
```
