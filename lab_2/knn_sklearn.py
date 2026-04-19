import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv('data/food_data.csv', sep=';', encoding='utf-8')
X = df[['сладость', 'хруст', 'калории', 'кислотность', 'жирность']]
y = df['класс']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

t0 = timeit.default_timer()
y_pred = knn.predict(X_test)
print(f"Время предсказания: {(timeit.default_timer() - t0) * 1000:.3f} ms")

print(f"Точность: {knn.score(X_test, y_test):.4f}")
print(f"\nОтчёт классификации:\n{classification_report(y_test, y_pred)}")

cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
print(f"Матрица ошибок:\n{cm}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel('Предсказанный класс')
plt.ylabel('Настоящий класс')
plt.title('Матрица ошибок — sklearn KNN')
plt.tight_layout()
plt.show()