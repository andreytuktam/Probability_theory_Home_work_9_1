import numpy as np 
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt


zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])
x = zp
y = ks

def mse_(y_pred, y = y, x = x, n = 10):
    return np.sum((B1 * x - y_pred) ** 2) / n

alpha = 1e-6
B1 = 0.1
n = 10

for i in range (6000):
    B1 -= alpha * (2 / n) * np.sum((B1 * x - y) * x)
    B0 = np.mean(x) - B1 * np.mean(y)
    y_pred = B0 + B1 * y
    if i % 600 == 0:
        print('iteration = {i}, y_pred = {y_pred} mse = {mse}'.format(i = i, y_pred = y_pred, mse = mse_(y_pred)))


