print("hello world")
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

st.set_page_config(page_title="CSV K-Means Explorer", layout="wide")

st.title("CSV K-Means Explorer")
st.write(
    "Upload a CSV file and run K-Means clustering. "
    "If you skip the upload, a sample dataset will be generated for you."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("No CSV uploaded. Generating a sample dataset.")
    sample_points = st.slider("Sample size", min_value=100, max_value=1000, value=300)
    sample_clusters = st.slider("Sample cluster count", min_value=2, max_value=6, value=3)
    data, _ = make_blobs(n_samples=sample_points, centers=sample_clusters, n_features=4, random_state=42)
    df = pd.DataFrame(data, columns=["feature_1", "feature_2", "feature_3", "feature_4"])

    # Add this here (download sample CSV)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download sample dataset (CSV)",
        data=csv_bytes,
        file_name="sample_dataset.csv",
        mime="text/csv",
    )

else:
    bytes_data = uploaded_file.getvalue()
    df = pd.read_csv(io.BytesIO(bytes_data))

st.subheader("Data preview")
st.dataframe(df.head(20), use_container_width=True)
api_key = st.sidebar.text_input("Your OpenAI API key", type="password")
st.sidebar.write.(api_key)

numeric_columns = df.select_dtypes(include="number").columns.tolist()

if len(numeric_columns) < 2:
    st.error("Please provide a CSV with at least two numeric columns.")
    st.stop()

selected_columns = st.multiselect(
    "Select numeric columns for clustering",
    options=numeric_columns,
    default=numeric_columns[:4],
)

if len(selected_columns) < 2:
    st.warning("Select at least two numeric columns to run K-Means.")
    st.stop()

k_value = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)

cluster_data = df[selected_columns].dropna()

if cluster_data.empty:
    st.error("Selected columns have no usable numeric data after removing missing values.")
    st.stop()

kmeans = KMeans(n_clusters=k_value, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(cluster_data)

cluster_results = cluster_data.copy()
cluster_results["cluster"] = cluster_labels

st.subheader("Clustered data")
st.dataframe(cluster_results.head(20), use_container_width=True)

st.subheader("Cluster visualization")

if cluster_data.shape[1] > 2:
    pca = PCA(n_components=2, random_state=42)
    projected = pca.fit_transform(cluster_data)
    plot_df = pd.DataFrame(projected, columns=["Component 1", "Component 2"])
    st.caption("Visualization uses PCA to reduce dimensions to 2 components.")
else:
    plot_df = cluster_data.iloc[:, :2].copy()
    plot_df.columns = ["Component 1", "Component 2"]

plot_df["cluster"] = cluster_labels

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    plot_df["Component 1"],
    plot_df["Component 2"],
    c=plot_df["cluster"],
    cmap="tab10",
    alpha=0.8,
)
legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.add_artist(legend)
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_title("K-Means Clusters")
ax.grid(True, linestyle="--", alpha=0.4)

st.pyplot(fig)

st.subheader("Cluster centers")
centers = pd.DataFrame(kmeans.cluster_centers_, columns=selected_columns)

st.dataframe(centers, use_container_width=True)

