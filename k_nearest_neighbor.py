# K Nearest Neighbor

# Algorithm to determine data class by calculating magnitudes
# of vectors representing that data on coord plane.

# STL
import warnings
import random
from collections import Counter
# Sci-computing lib (N-dimensional array)
import numpy as np
import pandas as pd

# It's better to set k to some odd number so theres
# couldn't be a situation with 50/50 result.
def k_nearest_neighbor(data, predict, k=3):
  if len(data) >= k:
    warnings.warn("K can't be less than total voting groups.")
  # Calculate magnitudes of vectors (distance from predict to other points).
  ms = []
  for group in data:
    for features in data[group]:
      # Formula for counting 2d vector's magnitude is
      # x^2 + y^2
      # Here's the same idea but np.linalg.norm is c-optimized.
      m = np.linalg.norm(np.array(features) - np.array(predict))
      ms.append([m, group])
  # Find most suitable result.
  votes = [i[1] for i in sorted(ms)[:k]]
  return Counter(votes).most_common(1)[0][0]

# Load data from file.
df = pd.read_csv('data/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

# Shuffle data and separate it for training and testing.
random.shuffle(full_data)
test_size = 0.2
# 2 - benign tumors. 4 - malignant tumors (just like in data file).
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

# Populate dictionaries.
for i in train_data:
  train_set[i[-1]].append(i[:-1])

for i in test_data:
  test_set[i[-1]].append(i[:-1])

# Testing.
correct = 0
total = 0

for group in test_set:
  for data in test_set[group]:
    vote = k_nearest_neighbor(train_set, data, k=5)
    if group == vote:
      correct += 1
    total += 1
print("Confidence = {}".format(correct / total))