import math
import random
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


def kmeans_clustering(K, lat, lon, n, max_iterations=100, tolerance=0.0001):
    centers_lat = [lat[i] for i in random.sample(range(n), K)]
    centers_lon = [lon[i] for i in random.sample(range(n), K)]
    clusters = {i: [] for i in range(K)}

    for _ in range(max_iterations):
        new_clusters = {i: [] for i in range(K)}

        for city_idx in range(n):
            nearest = min(range(K), key=lambda k: euclidean_dist(lat[city_idx], lon[city_idx], centers_lat[k], centers_lon[k]))
            new_clusters[nearest].append(city_idx)

        new_centers_lat, new_centers_lon = [], []
        for k in range(K):
            if not new_clusters[k]:
                new_centers_lat.append(centers_lat[k])
                new_centers_lon.append(centers_lon[k])
            else:
                new_centers_lat.append(sum(lat[i] for i in new_clusters[k]) / len(new_clusters[k]))
                new_centers_lon.append(sum(lon[i] for i in new_clusters[k]) / len(new_clusters[k]))

        moved = any(
            euclidean_dist(centers_lat[k], centers_lon[k], new_centers_lat[k], new_centers_lon[k]) > tolerance
            for k in range(K)
        )

        centers_lat, centers_lon, clusters = new_centers_lat, new_centers_lon, new_clusters

        if not moved:
            break

    return clusters, centers_lat, centers_lon


def compute_inertia(clusters, lat, lon):
    inertia = 0.0
    for cluster in clusters.values():
        if not cluster:
            continue
        c_lat = sum(lat[i] for i in cluster) / len(cluster)
        c_lon = sum(lon[i] for i in cluster) / len(cluster)
        inertia += sum(euclidean_dist(lat[i], lon[i], c_lat, c_lon) ** 2 for i in cluster)
    return inertia


# --- Автоматический выбор K методом локтя ---
k_values = list(range(1, min(10, n) + 1))
inertias = [
    min(compute_inertia(kmeans_clustering(k, lat, lon, n)[0], lat, lon) for _ in range(10))
    for k in k_values
]

kneedle = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
K = kneedle.elbow if kneedle.elbow is not None else 2
print(f"Автоматически выбрано K = {K}")

# --- Основной запуск с замером ---
tracemalloc.start()
start_time = time.perf_counter()

clusters, centers_lat, centers_lon = kmeans_clustering(K, lat, lon, n)

elapsed = time.perf_counter() - start_time
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

# --- Вывод результатов ---
print(f"\nРаспределение городов на {K} кластеров:\n")
for k in range(K):
    names = [cities[i] for i in clusters[k]]
    print(f"Кластер {k + 1}: {', '.join(names)} ({len(names)} городов)")

print(f"\nВремя выполнения: {elapsed:.6f} сек.")
print(f"Пиковая память: {peak_mem / 1024:.4f} KB")

# --- Силуэт ---
labels = [0] * n
for k, indices in enumerate(clusters.values()):
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
for k in range(K):
    indices = clusters[k]
    plt.scatter(
        [lon[i] for i in indices],
        [lat[i] for i in indices],
        c=[colors[k]], s=100, label=f'Кластер {k + 1}'
    )
    for i in indices:
        plt.annotate(cities[i], (lon[i], lat[i]), textcoords="offset points", xytext=(5, 5), fontsize=7)

plt.scatter(centers_lon, centers_lat, c='black', marker='X', s=200, zorder=5, label='Центроиды')
plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.title(f'K-means (K = {K})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()