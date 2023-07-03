import numpy as np
from operator import eq
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

centers = [[0, 1], [2, 4], [3, 2], [-2, 4], [-3, -1]]
n_classes = len(centers)

data, labels = make_blobs(n_samples=100,
                          centers=np.array(centers))

train_x, test_x, train_y, test_y = train_test_split(data, labels,
                                                    train_size=0.9,
                                                    test_size=0.1)
colours = ('red', 'green', 'blue', 'pink', 'black')

fig, ax = plt.subplots()
for n_class in range(0, n_classes):
    ax.scatter(train_x[train_y==n_class, 0], train_x[train_y==n_class, 1], c=colours[n_class], label=str(n_class))
ax.scatter(test_x[:,0], test_x[:,1], c=[[1,0,1]], label='test')
ax.legend(loc='upper left')

k = int(np.power(len(test_x), 1/3))
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(train_x, train_y)

predict = knn.predict(test_x)

accuracy = sum(map(eq, test_y, predict))/len(predict)*100

print(f'Предсказания классификатора:\n{predict}')
print(f'Реальные значения:\n{test_y}')
print(f'Процент правильных ответов: {accuracy}%')

for i, point in enumerate(test_x):
    ax.annotate(f'{test_y[i]}/{predict[i]}', (point[0]-0.1, point[1]-0.5))

plt.show()


