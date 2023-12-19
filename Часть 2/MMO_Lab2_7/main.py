import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# 1. Загрузить набор данных выявления аномалий
data = pd.read_csv('ec2_cpu_utilization_5f5533.csv')

# 2. Получить представление о наборе данных
print(data.shape)  # размерность данных
print(data.head())  # первые строки данных
print(data.describe())  # статистика данных
print(data.info())  # информация о данных
print(np.sum(data.isnull()))  # количество пустых значений в каждом признаке

# 3. Разведочный анализ данных
# Гистограмма распределения данных
plt.hist(data['value'], bins=30)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Value')
plt.show()

# График распределения данных
sns.kdeplot(data['value'], shade=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Distribution of Value')
plt.show()

# Кластеризация данных
kmeans = KMeans(n_clusters=2, random_state=0).fit(data[['value']])
data['cluster'] = kmeans.labels_

# Обучение моделей
# Метод k-средних (k-means clustering) уже обучен выше
# Цепь Маркова (Markov Chain) - не реализовано в sklearn
isolation_forest = IsolationForest().fit(data[['value']])
one_class_svm = OneClassSVM().fit(data[['value']])

# 7. Разведочный анализ данных для оценки качества обучения модели
# Визуализация результатов кластеризации
fig = px.scatter(data, x='timestamp', y='value', color='cluster')
fig.show()
