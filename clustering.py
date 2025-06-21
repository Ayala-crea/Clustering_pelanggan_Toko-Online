import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import os

# Load the dataset
file_path = "/mnt/data/dunia-anak-ceria_tokopedia_full_data.csv"
df = pd.read_csv(file_path)

# Tampilkan kolom yang tersedia
columns = df.columns

# Hitung jumlah pelanggan per kategori produk
# Asumsinya: setiap baris = 1 pelanggan (melalui ulasan)
# dan 'Kategori Produk' adalah kolom kategorinya

if 'Kategori Produk' in df.columns:
    customer_count_per_category = df.groupby('Kategori Produk').size().reset_index(name='Jumlah Pelanggan')

    # Encoding kategori produk menjadi angka
    encoder = LabelEncoder()
    customer_count_per_category['Kategori Produk (Encoded)'] = encoder.fit_transform(customer_count_per_category['Kategori Produk'])

    # Clustering jumlah pelanggan
    X = customer_count_per_category[['Jumlah Pelanggan']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_count_per_category['Cluster'] = kmeans.fit_predict(X)

    # Visualisasi
    plt.figure(figsize=(10, 6))
    sns.barplot(data=customer_count_per_category, x='Kategori Produk', y='Jumlah Pelanggan', hue='Cluster')
    plt.xticks(rotation=45, ha='right')
    plt.title('Clustering Jumlah Pelanggan Berdasarkan Kategori Produk')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    import ace_tools as tools; tools.display_dataframe_to_user(name="Hasil Clustering", dataframe=customer_count_per_category)
else:
    result = "Kolom 'Kategori Produk' tidak ditemukan dalam dataset."
