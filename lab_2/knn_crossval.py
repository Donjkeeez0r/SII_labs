import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv('data/food_data.csv', sep=';', encoding='utf-8')
X = df[['сладость', 'хруст', 'калории', 'кислотность', 'жирность']]
y = df['класс']

# объединения, чтобы нормализация применялась автоматически на каждом шаге
pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())

param_grid = {
    'kneighborsclassifier__n_neighbors': list(range(1, 11)),
    'kneighborsclassifier__metric': ['euclidean', 'manhattan']
}

# перебор всех комбинаций параметров (2 метрики - евкл и манх), выбирает лучшую комбинацию
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучшая точность (GridSearch): {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# разбиение датасета на 10 частей (9 для обучения, 1 для теста и по кругу)
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=k_fold, scoring='accuracy')

print(f"\nK-Fold CV scores: {cv_scores}")
print(f"Средняя точность: {cv_scores.mean():.4f}")

t0 = timeit.default_timer()
y_pred = cross_val_predict(best_model, X, y, cv=k_fold)
print(f"Время предсказания: {(timeit.default_timer() - t0) * 1000:.3f} ms")

print(f"\nОтчёт классификации:\n{classification_report(y, y_pred)}")

cm = confusion_matrix(y, y_pred, labels=sorted(y.unique()))
print(f"Матрица ошибок:\n{cm}")

# График матрицы ошибок
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel('Предсказанный класс')
plt.ylabel('Настоящий класс')
plt.title('Матрица ошибок — кросс-валидация')
plt.tight_layout()
plt.show()

# График важности признаков
importance = permutation_importance(best_model, X, y, n_repeats=10, random_state=42).importances_mean
plt.figure(figsize=(7, 4))
plt.bar(X.columns, importance, color='steelblue')
plt.xlabel('Признак')
plt.ylabel('Важность')
plt.title('Важность признаков (Permutation Importance)')
plt.tight_layout()
plt.show()