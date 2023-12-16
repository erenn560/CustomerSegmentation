# Gerekli kütüphaneleri yükle
import numpy as np
import os
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Çalışma ortamında thread sayısını belirt
os.environ["OMP_NUM_THREADS"] = "1"

# CSV dosyasını oku
df = pd.read_csv("Avm_Musterileri.csv")
df.head()

# Veriyi görselleştir: Annual Income (gelir) ve Spending Score (harcama puanı) arasındaki dağılımı gösteren bir scatter plot
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Veri çerçevesindeki sütun adlarını yeniden adlandır
df.rename(columns={'Annual Income (k$)': 'income'}, inplace=True)
df.rename(columns={'Spending Score (1-100)': 'score'}, inplace=True)

# Veriyi normalize et: MinMaxScaler kullanarak her iki özelliği [0, 1] aralığına ölçekle
scaler = MinMaxScaler()
scaler.fit(df[['income']])
df['income'] = scaler.transform(df[['income']])
scaler.fit(df[['score']])
df['score'] = scaler.transform(df[['score']])
df.head()

# KMeans algoritması için kümelerin sayısını belirlemek için bir 'Elbow Method' grafiği oluştur
k_range = range(1, 11)
list_dist = []

for k in k_range:
    kmeans_model = KMeans(n_clusters=k, n_init=10)
    kmeans_model.fit(df[['income', 'score']])
    list_dist.append(kmeans_model.inertia_)

# Elbow Method grafiği
plt.xlabel('K')
plt.ylabel('Distortion değeri (inertia)')
plt.plot(k_range, list_dist)
plt.show()

# KMeans modelini oluştur ve veriyi kümelere ayır
kmeans_model = KMeans(n_clusters=5, n_init=10)
y_predicted = kmeans_model.fit_predict(df[['income', 'score']])
y_predicted

# Oluşan kümeleri DataFrame'e ekle
df['cluster'] = y_predicted
df.head()

# Küme merkezlerini al
kmeans_model.cluster_centers_

# Her kümeyi ayrı bir renk ve şekilde gösteren bir scatter plot oluştur
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]
df4 = df[df.cluster == 3]
df5 = df[df.cluster == 4]

plt.xlabel('income')
plt.ylabel('score')
plt.scatter(df1['income'], df1['score'], color='blue')
plt.scatter(df2['income'], df2['score'], color='yellow')
plt.scatter(df3['income'], df3['score'], color='black')
plt.scatter(df4['income'], df4['score'], color='orange')
plt.scatter(df5['income'], df5['score'], color='grey')

# Küme merkezlerini belirgin hale getirerek grafiğe ekle
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], color='blue', marker='X',
            label='centroid')
plt.legend()
plt.show()
