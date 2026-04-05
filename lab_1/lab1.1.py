import math
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

df = pd.read_csv('tests/main_100.csv', sep=';', encoding='utf-8')
cities = df['City'].tolist()
lat = df['Latitude'].tolist()
lon = df['Longitude'].tolist()
n = len(cities)


def euclidean_dist(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


def build_distance_matrix(n, lat, lon):
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_dist(lat[i], lon[i], lat[j], lon[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def hierarchical_clustering(K, dist_matrix, n):
    clusters = [[i] for i in range(n)]

    while len(clusters) > K:
        min_dist = float('inf')
        best_pair = (-1, -1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                max_dist = max(
                    dist_matrix[a][b]
                    for a in clusters[i]
                    for b in clusters[j]
                )
                if max_dist < min_dist:
                    min_dist = max_dist
                    best_pair = (i, j)

        i, j = best_pair
        clusters[i] += clusters.pop(j)

    return clusters


def compute_inertia(clusters, lat, lon):
    inertia = 0.0
    for cluster in clusters:
        if not cluster:
            continue
        c_lat = sum(lat[i] for i in cluster) / len(cluster)
        c_lon = sum(lon[i] for i in cluster) / len(cluster)
        inertia += sum(euclidean_dist(lat[i], lon[i], c_lat, c_lon) ** 2 for i in cluster)
    return inertia


dist_matrix = build_distance_matrix(n, lat, lon)

# --- Автоматический выбор K методом локтя ---
k_values = list(range(1, min(10, n) + 1))
inertias = [compute_inertia(hierarchical_clustering(k, dist_matrix, n), lat, lon) for k in k_values]

kneedle = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
K = kneedle.elbow if kneedle.elbow is not None else 2
print(f"Автоматически выбрано K = {K}")

# --- Основной запуск с замером ---
tracemalloc.start()
start_time = time.perf_counter()

clusters = hierarchical_clustering(K, dist_matrix, n)

elapsed = time.perf_counter() - start_time
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

# --- Вывод результатов ---
print(f"\nРаспределение городов на {K} кластеров:\n")
for k, indices in enumerate(clusters):
    print(f"Кластер {k + 1}: {', '.join(cities[i] for i in indices)} ({len(indices)} городов)")

print(f"\nВремя выполнения: {elapsed:.6f} сек.")
print(f"Пиковая память: {peak_mem / 1024:.4f} KB")

# --- Силуэт ---
labels = [0] * n
for k, indices in enumerate(clusters):
    for i in indices:
        labels[i] = k

features = np.array(list(zip(lat, lon)))
print(f"Коэффициент силуэта: {silhouette_score(features, labels):.4f}")

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
for k, indices in enumerate(clusters):
    plt.scatter(
        [lon[i] for i in indices],
        [lat[i] for i in indices],
        c=[colors[k]], s=100, label=f'Кластер {k + 1}'
    )
    for i in indices:
        plt.annotate(cities[i], (lon[i], lat[i]), textcoords="offset points", xytext=(5, 5), fontsize=7)

plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.title(f'Иерархическая кластеризация (K = {K})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()