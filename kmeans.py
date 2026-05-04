import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Clustering Customer", layout="centered")

st.title("📊 Aplikasi Clustering Customer (K-Means)")
st.write("Upload dataset atau gunakan dataset default")

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
else:
    dataset = pd.read_csv("Mall_Customers.csv")

st.subheader("Dataset")
st.dataframe(dataset.head())

# =========================
# PILIH FITUR
# =========================
st.subheader("Pilih Fitur")

col1 = st.selectbox("Pilih fitur X", dataset.columns, index=3)
col2 = st.selectbox("Pilih fitur Y", dataset.columns, index=4)

X = dataset[[col1, col2]]

if col1 == col2:
    st.error("Fitur X dan Y tidak boleh sama!")
    st.stop()
# =========================
# ELBOW METHOD
# =========================
st.subheader("Elbow Method")

max_k = st.slider("Max jumlah cluster", 2, 10, 10)

wcss = []
for i in range(1, max_k+1):
    kmeans = KMeans(n_clusters=i, random_state=14)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1, max_k+1), wcss, marker='o')
ax1.set_title("Elbow Method")
ax1.set_xlabel("Jumlah Cluster")
ax1.set_ylabel("WCSS")
st.pyplot(fig1)

# =========================
# PILIH K
# =========================
k = st.slider("Pilih jumlah cluster (k)", 2, 10, 5)

# =========================
# KMEANS
# =========================
kmeans = KMeans(n_clusters=k, random_state=14)
y_pred = kmeans.fit_predict(X)

hasil = X.copy()
hasil["Cluster"] = y_pred

# =========================
# VISUALISASI
# =========================
st.subheader("Visualisasi Clustering")

fig2, ax2 = plt.subplots()

for i in range(k):
    ax2.scatter(
        hasil[hasil["Cluster"] == i][col1],
        hasil[hasil["Cluster"] == i][col2],
        label=f"Cluster {i}"
    )

# centroid
centroid = kmeans.cluster_centers_
ax2.scatter(centroid[:, 0], centroid[:, 1],
            s=200, c='black', label="Centroid")

ax2.set_xlabel(col1)
ax2.set_ylabel(col2)
ax2.legend()

st.pyplot(fig2)

# =========================
# HASIL DATA
# =========================
st.subheader("Hasil Clustering")
st.dataframe(hasil)

# =========================
# DOWNLOAD
# =========================
csv = hasil.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download Hasil CSV",
    csv,
    "hasil_clustering.csv",
    "text/csv"
)