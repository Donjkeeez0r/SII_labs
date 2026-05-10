import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import ARDRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# загрузка данных
train = pd.read_csv('data/train.csv', sep='\t', header=None)
test = pd.read_csv('data/test.csv', sep='\t', header=None)

feature_names = [f'X{i + 1}' for i in range(train.shape[1] - 1)]
train.columns = feature_names + ['Y']
test.columns = feature_names

X = train[feature_names]
y = train['Y']

# добавление шума (для проверки)
# noise = np.random.normal(0, 0.05, size=y.shape)
# y = y + noise

# добавление выбросов
# outlier_idx = np.random.choice(len(y), size=50, replace=False)
# y.iloc[outlier_idx] = y.iloc[outlier_idx] * 5
#

X_test = test[feature_names]
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print('Размер train:', train.shape)
print('Размер test:', test.shape)
print('\nПервые строки train:')
print(train.head())


# делим train на обучающую и проверочную части
X_train, X_val, y_train, y_val = train_test_split(
	X, y, test_size=0.2, random_state=42
)


# модель
model = ARDRegression()
model.fit(X_train, y_train)


# оценка
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

mae_train = mean_absolute_error(y_train, pred_train)
mae_val = mean_absolute_error(y_val, pred_val)
rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
r2_train = r2_score(y_train, pred_train)
r2_val = r2_score(y_val, pred_val)

print('\nКачество на train:')
print(f'MAE: {mae_train:.6f}')
print(f'RMSE: {rmse_train:.6f}')
print(f'R2: {r2_train:.6f}')

print('\nКачество на validation:')
print(f'MAE: {mae_val:.6f}')
print(f'RMSE: {rmse_val:.6f}')
print(f'R2: {r2_val:.6f}')


# предсказания для test
final_model = ARDRegression()
final_model.fit(X, y)
test_pred = final_model.predict(X_test)
print('\nПервые предсказания для test:')
print(np.round(test_pred[:10], 6))


# графики
plt.figure(figsize=(6, 5))
plt.scatter(y_val, pred_val, alpha=0.5, color='steelblue')
mn = min(y_val.min(), pred_val.min())
mx = max(y_val.max(), pred_val.max())
plt.plot([mn, mx], [mn, mx], 'r--')
plt.xlabel('Настоящая цена')
plt.ylabel('Предсказанная цена')
plt.title('Факт vs предсказание')
plt.tight_layout()
plt.show()

residuals = y_val - pred_val

plt.figure(figsize=(6, 5))
sns.histplot(residuals, bins=30, kde=True, color='orange')
plt.xlabel('Ошибка')
plt.ylabel('Количество')
plt.title('Распределение ошибок')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(pred_val, residuals, alpha=0.5, color='green')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Предсказание')
plt.ylabel('Ошибка')
plt.title('Ошибки от предсказаний')
plt.tight_layout()
plt.show()


# важность признаков
coef = final_model.coef_
top_idx = np.argsort(np.abs(coef))[-15:]

plt.figure(figsize=(10, 5))
plt.bar([feature_names[i] for i in top_idx], coef[top_idx], color='steelblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Признаки')
plt.ylabel('Коэффициенты')
plt.title('Важность признаков')
plt.tight_layout()
plt.show()