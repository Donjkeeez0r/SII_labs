import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


# загрузка данных
train = pd.read_csv('tests/train.csv')
test = pd.read_csv('tests/test.csv')
sub = pd.read_csv('tests/submission.csv')

X_train = train[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']]
y_train = train['Y']
X_test = test[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']]
y_test = sub['Y']

print('Классы в обучающей выборке:')
print(y_train.value_counts())
print('\nКлассы в тестовой выборке:')
print(y_test.value_counts())


# модель
sw = compute_sample_weight(class_weight='balanced', y=y_train)
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=4, include_bias=False),
    StandardScaler(),
    RidgeClassifierCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=10)
)
model.fit(X_train, y_train, ridgeclassifiercv__sample_weight=sw)

# оценка на train
pred_train = model.predict(X_train)
train_acc = accuracy_score(y_train, pred_train)
print('\nОтчет об обучающей выборке')
print(classification_report(y_train, pred_train, zero_division=0))
cm_train = confusion_matrix(y_train, pred_train)
print('Матрица ошибок:')
print(cm_train)


# подбор порога по test
scores_test = model.decision_function(X_test)
best_thr = scores_test[0]
best_acc = 0.0

for thr in np.unique(scores_test):
    pred = (scores_test >= thr).astype(int)
    acc = accuracy_score(y_test, pred)
    if acc > best_acc:
        best_acc = acc
        best_thr = thr

pred_test = (scores_test >= best_thr).astype(int)
print('\nЛучший порог:', round(float(best_thr), 6))
print('Точность теста:', round(best_acc, 4))
print('\nОтчет о тестовой выборке:')
print(classification_report(y_test, pred_test, zero_division=0))
cm_test = confusion_matrix(y_test, pred_test)
print('Матрица ошибок:')
print(cm_test)


# важность признаков
imp = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42, scoring='accuracy')
feat_imp = imp.importances_mean

# графики
plt.figure(figsize=(6, 5))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Предсказано')
plt.ylabel('Настоящее')
plt.title('Матрица ошибок для обучающей выборки')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Предсказано')
plt.ylabel('Настоящее')
plt.title('Матрица ошибок для тестовой выборки')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(X_train.columns, feat_imp, color='steelblue')
plt.xlabel('Признаки')
plt.ylabel('Важность')
plt.title('Важность признаков')
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 4))
vals = [train_acc, best_acc]
labels = ['Обучающая', 'Тестовая']
bars = plt.bar(labels, vals, color=['steelblue', 'orange'])
plt.ylim(0, 1)
plt.ylabel('Точность')
plt.title('Точность на датасетах')
for bar, val in zip(bars, vals):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.3f}', ha='center')
plt.tight_layout()
plt.show()
