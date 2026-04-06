Отчет 1: https://docs.google.com/document/d/1cGEXgIlGf12CBdkdmDsR2E2kKV_XbZUHkEQOL15G7K4/edit?usp=sharing

import time
import tracemalloc
import random
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Генерация 100 случайных городов
n = 100
random.seed(42)  # Фиксируем seed для воспроизводимости результатов

cities = [f"Город_{i+1}" for i in range(n)]
# Генерация случайных координат: широта от -90 до 90, долгота от -180 до 180
latitudes = [random.uniform(-90, 90) for _ in range(n)]
longitudes = [random.uniform(-180, 180) for _ in range(n)]

# Формируем массив признаков
features = np.array(list(zip(latitudes, longitudes)))

print(f"Сгенерировано {n} случайных городов")

# --- Автоматический выбор K методом локтя ---
k_values = list(range(1, min(10, n) + 1))
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(features).inertia_ for k in k_values]

kneedle = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
K = kneedle.elbow if kneedle.elbow is not None else 2
print(f"Автоматически выбрано K = {K}")

# --- Основной запуск с замером ---
tracemalloc.start()
start_time = time.perf_counter()

kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features)

elapsed = time.perf_counter() - start_time
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

centers = kmeans.cluster_centers_

# --- Вывод результатов ---
print(f"\nРаспределение городов на {K} кластеров:\n")
for k in range(K):
    names = [cities[i] for i in range(n) if cluster_labels[i] == k]
    # Показываем первые 5 городов для наглядности
    display_names = names[:5]
    if len(names) > 5:
        display_names.append(f"... и ещё {len(names) - 5}")
    print(f"Кластер {k + 1}: {', '.join(display_names)} ({len(names)} городов)")

print(f"\nВремя выполнения: {elapsed:.6f} сек.")
print(f"Пиковая память: {peak_mem / 1024:.4f} KB")
print(f"Коэффициент силуэта: {silhouette_score(features, cluster_labels):.4f}")

# --- График локтя ---
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=K, color='r', linestyle='--', label=f'Оптимальное K = {K}')
plt.xlabel('Количество кластеров (K)')
plt.ylabel('Инерция')
plt.title('Метод локтя')
plt.xticks(k_values)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- График кластеров ---
colors = plt.cm.tab10(np.linspace(0, 1, K))
plt.figure(figsize=(8, 6))
for k in range(K):
    cluster_indices = [i for i in range(n) if cluster_labels[i] == k]
    cluster_lats = [latitudes[i] for i in cluster_indices]
    cluster_lons = [longitudes[i] for i in cluster_indices]
    
    plt.scatter(cluster_lons, cluster_lats,
                c=[colors[k]], s=100, label=f'Кластер {k + 1}')
    
    # Подписываем только первые 3 города в каждом кластере
    for i in cluster_indices[:3]:
        plt.annotate(cities[i], (longitudes[i], latitudes[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

plt.scatter(centers[:, 1], centers[:, 0], c='black', marker='X', s=200, zorder=5, label='Центроиды')
plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.title(f'K-means кластеризация случайных городов (K = {K})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
