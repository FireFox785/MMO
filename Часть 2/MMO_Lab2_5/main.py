import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV

# 1. Загрузить набор данных
data = pd.read_csv('kc_house_data.csv')

# 2. Получить представление о наборе данных
print(data.shape)
print(data.head())
print(data.describe())
print(data.info())
print(data.isnull().sum())

# 3. Разведочный анализ данных
# Гистограммы
data.hist(figsize=(10,10))
plt.show()

# Ящик с усами (диаграмма размаха)
sns.boxplot(x=data['feature'], y=data['price'])
plt.show()

# Точечные 3D-графики распределения признаков
fig = px.scatter_3d(data, x='feature1', y='feature2', z='price')
fig.show()

# Тепловая карта корреляционной матрицы
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# 4. Выводы о влиянии характеристик на стоимость недвижимости

# 5. Объединить значения по периодам и произвести нормализацию данных
# (предположим, что признаки для даты постройки и ремонта уже существуют в данных)

# Нормализация данных

# Разбить обработанный набор данных на обучающую и тестовую выборки
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Обучение моделей
# Линейная регрессия
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Регрессия LASSO
lasso_reg = Lasso()
lasso_reg.fit(X_train, y_train)

# Регрессия Ridge
ridge_reg = Ridge()
ridge_reg.fit(X_train, y_train)

# Полиномиальная регрессия
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# 7. Отображение точности работы моделей
models = [lin_reg, lasso_reg, ridge_reg, poly_reg]
for model in models:
    y_pred = model.predict(X_test)
    print(model)
    print('R-squared:', r2_score(y_test, y_pred))
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

# 8. Метрики после кросс-валидации
for model in models:
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(model)
    print('Cross-validated R-squared:', np.mean(scores))

# 9. Поиск оптимальных гиперпараметров
param_grid = {'alpha': [0.1, 1, 10]}
lasso_grid = GridSearchCV(lasso_reg, param_grid)
lasso_grid.fit(X_train, y_train)
print(lasso_grid.best_params_)