import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# 1. Загрузка набора данных
df = pd.read_csv('DailyDelhiClimateTrain.csv')

# 2. Представление о наборе данных
print(df.shape)
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())

# 3. Разведочный анализ данных
# Гистограммы распределения данных
df.hist(figsize=(10,10))
plt.show()

# Графики распределения данных
sns.pairplot(df)
plt.show()

# Ящики с усами (диаграммы размаха)
fig = px.box(df, y="Temperature")
fig.show()

# 4. Выводы об особенностях набора данных
# Набор данных содержит информацию о погодных условиях, включая температуру, влажность и давление.

# 5. Кластеризация данных
# Необходимо выбрать признаки для кластеризации и применить соответствующий метод кластеризации (например, KMeans).

# 6. Обучение модели XGBoost Regressor
X = df.drop('Temperature', axis=1)
y = df['Temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()
model.fit(X_train, y_train)

# 7. Отображение точности работы модели
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("R-squared on training set:", r2_train)
print("R-squared on test set:", r2_test)