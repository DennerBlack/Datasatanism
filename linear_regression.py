import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('LinearRegressionData2.csv')

data_len = 200
synth_data_sin = {'X': np.arange(data_len),
                  'Y': np.sin(np.arange(data_len)*np.random.uniform(0.05, 0.15))+np.random.normal(0, 0.03, data_len)}
#                                                 ^ частота(ширина) синусоиды    ^ добавление шумов  ^ величина шума
synth_data_linear = {'X': np.arange(data_len),
                     'Y': np.arange(data_len)+np.random.normal(0, np.log(data_len), data_len)}
#                                             ^ добавление шумов  ^ величина шума


def calculate_slope(x, y):
    mx = x - x.mean()
    my = y - y.mean()
    return sum(mx * my) / sum(mx**2)


def get_params(x, y):
    a = calculate_slope(x, y)
    b = y.mean() - a * x.mean()
    return a, b


def get_correlation_coeff(x, y):
    cov = sum((x-np.mean(x))*(y-np.mean(y)))/(len(x)-1)
    R = cov/np.sqrt(sum(np.power(x-np.mean(x), 2))*sum(np.power(y-np.mean(y), 2)))*(len(x)-1)
    return R


d = synth_data_linear
x = d['X']
y = d['Y']
a, b = get_params(x, y)

R = get_correlation_coeff(x, y)
print(f'Коэффициент корреляции Пирсона {R=:.3f}')
if np.abs(R) >= 0.7:
    print('Линейная корреляция явно выражена')
else:
    print('Слабая или отсутствующая линейная корреляция')

lin_reg = a*x + b
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x, y)
plt.plot(x, lin_reg, color='red')
plt.show()


