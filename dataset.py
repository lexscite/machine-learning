# Dataset.

# Collection of instructions to manipultate data sets.

import random
# Sci-computing lib (N-dimensional array).
import numpy as np

# Generates some random data set.
def generate_random_dataset(size, variance, step=2, correlation=False):
  val = 1
  ys = []
  for i in range(size):
    y = val + random.randrange(-variance, variance)
    ys.append(y)
    if correlation and correlation == 'pos':
      val += step
    elif corellation and correlation == 'neg':
      val -= step
  xs = [i for i in range(len(ys))]
  return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)