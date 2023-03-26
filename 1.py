# Даны значения величины заработной платы заемщиков банка (zp) и значения 
# их поведенческого кредитного скоринга 
# (ks): zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110], 
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]. 
# Используя математические операции, посчитать коэффициенты линейной регрессии, 
# приняв за X заработную плату (то есть, zp - признак), а за y - значения 
# скорингового балла (то есть, ks - целевая переменная). Произвести расчет 
# как с использованием intercept, так и без.

import numpy as np 
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt


zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

x = zp.reshape((10, 1))
print(x)
y = ks.reshape((10, 1))
print(y)

B = np.dot(np.linalg.inv(np.dot(x.T, x)), x.T @ y)
print('without intercept', B)

X = np.hstack([np.ones((10, 1)),x])
print(X)

B_i = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T @ y)
print('intercept',B_i)


# b1 = (np.mean(zp * ks) - np.mean(zp) * np.mean(ks)) / (np.mean(zp ** 2) - np.mean(zp) ** 2)
# print(b1)
# b0 = np.mean(ks) - b1 * np.mean(zp) #интерсепт
# print(b0)
# y_pred = b0 + b1 * zp
# print(y_pred)

# plt.scatter(x_zp, y_ks)
# plt.title('r = вписать значение коэффициента')
# plt.xlabel('x_zp')
# plt.ylabel('y_ks')
# plt.plot(x_zp, y_pred)
# plt.show()
