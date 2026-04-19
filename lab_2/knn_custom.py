import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    '''
    для каждого объекта вычисляется евклидово расстояние до всех точек обучающей выборки
    затем берутся k ближ соседей, и потом класс определяется
    '''
    def predict(self, X):
        predictions = []
        for x in np.asarray(X):
            distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
            k_nearest_labels = [self.y_train[i] for i in np.argsort(distances)[:self.k]]
            predictions.append(Counter(k_nearest_labels).most_common(1)[0][0])
        return predictions

'''
читается датасет, разбивается 80 на 20
затем применяется StandardScaler для нормализации признаков
'''
df = pd.read_csv('data/food_data.csv', sep=';', encoding='utf-8')
X = df[['сладость', 'хруст', 'калории', 'кислотность', 'жирность']]
y = df['класс']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = KNN(k=5)
clf.fit(X_train, y_train)

t0 = timeit.default_timer()
predictions = clf.predict(X_test)
print(f"Время предсказания: {(timeit.default_timer() - t0) * 1000:.3f} ms")

print(f"Точность: {np.mean(np.asarray(predictions) == np.asarray(y_test)):.4f}")
print(f"\nОтчёт классификации:\n{classification_report(y_test, predictions)}")

cm = confusion_matrix(y_test, predictions, labels=sorted(y.unique()))
print(f"Матрица ошибок:\n{cm}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel('Предсказанный класс')
plt.ylabel('Настоящий класс')
plt.title('Матрица ошибок — собственный KNN')
plt.tight_layout()
plt.show()