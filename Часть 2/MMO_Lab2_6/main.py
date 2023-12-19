import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Загрузка набора данных
data = pd.read_csv('marketing_campaign.csv')

# 2. Получение информации о данных
print("Shape:", data.shape)
print("Head:", data.head())
print("Describe:", data.describe())
print("Info:", data.info())
print("Number of empty values:", np.sum(data.isnull()))

# 3. Визуализация данных
# Графики распределения признаков
data.hist(figsize=(12, 10))
plt.show()

# Тепловая карта корреляционной матрицы
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# 4. Выводы о распределении индивидуальных качеств покупателей
# Из графиков распределения признаков можно сделать выводы о средних значениях, диапазонах и форме распределения каждого признака.
# Из тепловой карты корреляционной матрицы можно сделать выводы о взаимосвязи между признаками.

# 5. Кодирование категориальных данных и стандартизация данных
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Age'] = label_encoder.fit_transform(data['Age'])
data['City_Category'] = label_encoder.fit_transform(data['City_Category'])
data['Stay_In_Current_City_Years'] = label_encoder.fit_transform(data['Stay_In_Current_City_Years'])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 6. Снижение размерности набора данных
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# 7. Обучение модели методом k-средних
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(reduced_data)

# 8. Разведочный анализ данных по кластерам
cluster_labels = kmeans.labels_
data['Cluster'] = cluster_labels

# Визуализация кластеров
fig = px.scatter(data, x='PC1', y='PC2', color='Cluster')
fig.show()