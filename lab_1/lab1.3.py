import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('tests/main_100.csv', sep=';', encoding='utf-8')
features = df[['Latitude', 'Longitude']].values
cities = df['City'].tolist()
n = len(cities)

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
df['Cluster_ID'] = kmeans.fit_predict(features)

elapsed = time.perf_counter() - start_time
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

centers = kmeans.cluster_centers_

# --- Вывод результатов ---
print(f"\nРаспределение городов на {K} кластеров:\n")
for k in range(K):
    names = df[df['Cluster_ID'] == k]['City'].tolist()
    print(f"Кластер {k + 1}: {', '.join(names)} ({len(names)} городов)")

print(f"\nВремя выполнения: {elapsed:.6f} сек.")
print(f"Пиковая память: {peak_mem / 1024:.4f} KB")
print(f"Коэффициент силуэта: {silhouette_score(features, df['Cluster_ID'].values):.4f}")

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
    cluster_df = df[df['Cluster_ID'] == k]
    plt.scatter(cluster_df['Longitude'], cluster_df['Latitude'],
                c=[colors[k]], s=100, label=f'Кластер {k + 1}')
    for _, row in cluster_df.iterrows():
        plt.annotate(row['City'], (row['Longitude'], row['Latitude']),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

plt.scatter(centers[:, 1], centers[:, 0], c='black', marker='X', s=200, zorder=5, label='Центроиды')
plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.title(f'K-means sklearn (K = {K})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()