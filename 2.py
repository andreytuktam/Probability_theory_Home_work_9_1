import numpy as np 
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt


zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])
x = zp
y = ks

def mse_(B1, y = y, x = x, n = 8):
    return np.sum((B1 * x - y) ** 2) / n
alpha = 1e-6

B1 = 0.1
n = 10

for i in range (6000):
    B1 -= alpha * (2 / n) * np.sum((B1 * x - y) * x)
    if i % 600 == 0:
        print('iteration = {i}, B1 = {B1}, mse = {mse}'.format(i = i, B1 = B1, mse = mse_(B1)))

print(mse_(5.889820))
