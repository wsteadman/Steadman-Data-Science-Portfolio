import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_breast_cancer, load_iris, load_wine


# -----------------------------------------------
# Helper Functions
# -----------------------------------------------


#Loading sample datasets from sklearn
def load_sample_dataset(dataset_name):
    if dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    elif dataset_name == "Iris":
        data = load_iris()
    elif dataset_name == "Wine":
        data = load_wine()
    
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y 

#Standardize features 
def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

#Run the KMeans ML model
def run_kmeans(X, k):
    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(X)
    return kmeans, clusters


#Compute data for elbow method and silhouette score
def compute_elbow_data(X, max_k=10):
    wcss = []
    silhouette_scores = []
    ks = range(2, max_k + 1)
    
    for i in ks:
        km = KMeans(n_clusters=i)
        km.fit(X)
        wcss.append(km.inertia_)
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X, labels))
    return ks, wcss, silhouette_scores

#Apply PCA
def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_


#Plot clusters using PCA
def plot_clusters(X_pca, clusters):
    plt.figure(figsize=(8, 6))
    
    unique_clusters = np.unique(clusters)
    colors = ['navy', 'darkorange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i in range(len(unique_clusters)):
        cluster = unique_clusters[i]
        plt.scatter(
            X_pca[clusters == cluster, 0],
            X_pca[clusters == cluster, 1],
            c=colors[i % len(colors)],
            alpha=0.7,
            edgecolor='k',
            s=60,
            label=f'Cluster {cluster}'
        )
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering: 2D PCA Projection')
    plt.legend(loc='best')
    plt.grid(True)
    
    fig = plt.gcf()
    return fig
    

#Create elbow method and silhouette score plots"""
def plot_elbow_method(ks, wcss, silhouette_scores):
    plt.figure(figsize=(12, 5))

    # Elbow Method
    plt.subplot(1, 2, 1)
    plt.plot(ks, wcss, 'o-', color='blue')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.title('Elbow Method')
    plt.grid(True)

    # Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(ks, silhouette_scores, 'o-', color='green')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    plt.grid(True)

    plt.tight_layout()
    fig = plt.gcf()  # Get the current figure
    return fig


# -----------------------------------------------
# Main App Layoutt
# -----------------------------------------------

# Set page configuration
st.set_page_config(
    page_title="KMeans Clustering Explorer",
    page_icon="📊",
)

# Title and introduction
st.title("📊 KMeans Clustering Explorer")
st.markdown("""
This app allows you to explore KMeans clustering on various datasets.
Upload your own dataset or use one of the sample datasets to discover natural groups in your data.
""")

# Sidebar for dataset selection and parameters
st.sidebar.header("Settings")

# Dataset selection
dataset_option = st.sidebar.radio(
    "Choose a dataset:",
    ["Breast Cancer", "Iris", "Wine", "Upload Your Own"]
)

# Load data based on selection
if dataset_option == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
            
            df = pd.read_csv(uploaded_file)
            
            # Allow user to select target column (optional)
            target_col = st.sidebar.selectbox(
                "Select target column (optional):",
                ["None"] + df.columns.tolist()
            )
            
            if target_col != "None":
                feature_cols = [col for col in df.columns if col != target_col]
                X = df[feature_cols]
                y = df[target_col]
    
            else:
                X = df
                y = None
                
    else:
        st.info("👈 Please upload a CSV file or select a sample dataset.")
        st.stop()
else:
    X, y = load_sample_dataset(dataset_option)

# Display dataset information
st.subheader("Dataset Overview")
st.write(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

st.markdown("""
- **Samples**: the individual data points or rows in your dataset
- **Features**: are the attributes or columns describing each sample (used to cluster rows together)
""")

# Display the first few rows of the dataset
with st.expander("Preview Dataset"):
    st.dataframe(X.head(10))
    st.write("Statistical Summary")
    st.dataframe(X.describe())

# Display target variable information after the dataset overview
if y is not None:
    st.markdown("#### Target Variable Information")
    
    # Get target information based on dataset
    target_info = ""
    if dataset_option == "Breast Cancer":
        target_info = "The target variable represents tumor diagnosis (0 = malignant, 1 = benign)."
    elif dataset_option == "Iris":
        target_info = "The target variable represents flower species (0 = setosa, 1 = versicolor, 2 = virginica)."
    elif dataset_option == "Wine":
        target_info = "The target variable represents different wine cultivars (0, 1, 2)."
    elif dataset_option == "Upload Your Own":
        target_info = f"Using '{target_col}' as the target variable."
    
    st.write(target_info)

# KMeans parameters
st.sidebar.header("KMeans Parameters")
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10)
scaling = st.sidebar.checkbox("Scale features", value=True)

# Run clustering when requested
if st.sidebar.button("Run Clustering"):
    # Scale features if selected and notify user
    if scaling:
        X_processed = scale_features(X)
        st.success("Data scaled")
    else:
        X_processed = X.values
        st.warning("Using unscaled data. KMeans works best with scaled features.")
    
    # Run KMeans
    kmeans_model, cluster_labels = run_kmeans(X_processed, k)

    # Add cluster labels to the original data
    clustered_data = X.copy()
    clustered_data['Cluster'] = cluster_labels


    # Extract target values from y regardless of whether it's a pandas Series (has .values attribute)
    # or already a NumPy array/list (doesn't have .values attribute)
    if y is not None:
        clustered_data['True_Label'] = y.values if hasattr(y, 'values') else y

    # Move these columns to the front
    clustered_data = clustered_data[['Cluster', 'True_Label'] + [col for col in clustered_data.columns if col not in ['Cluster', 'True_Label']]]

    # Create tabs for different analysis
    tab1, tab2, tab3 = st.tabs(["Cluster Visualization", "Optimal K Analysis", "Cluster Data"])
    
    with tab1:
        st.subheader("2D Visualization of Clusters")
        
        # Apply PCA 
        X_pca, explained_variance = apply_pca(X_processed)
        st.write(f"Explained variance: {sum(explained_variance):.2%}")
        
        # Plot clusters
        fig = plot_clusters(X_pca, cluster_labels)
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Finding the Optimal Number of Clusters")
        
        # Compute elbow method data
        ks, wcss, silhouette_scores = compute_elbow_data(X_processed)
        
        # Plot elbow method
        fig = plot_elbow_method(ks, wcss, silhouette_scores)
        st.pyplot(fig)
        
        st.info("""
        **How to interpret:**
        - **Elbow Method**: Look for the point where the decrease in WCSS slows down significantly.
        - **Silhouette Score**: Higher values indicate better-defined clusters.
        """)
        
    with tab3:
        st.subheader("Clustered Data Results")
        
        # Calculate and display cluster sizes
        st.write("### Cluster Sizes")
        cluster_counts = clustered_data['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)
        
        # Display Dataframe w cluster and target labels
        st.write("### Data with Cluster Assignments")
        st.dataframe(clustered_data)

        st.markdown("""
            - **Note:** When using a sample dataset like *Breast Cancer*, the original class labels (e.g. malignant vs. benign) are available and shown as `True_Label`.  
            This allows you to compare the clustering results (`Cluster`) to the actual known classes (`True_Label`).  
            Keep in mind that KMeans doesn’t know these labels — it just groups similar data points based on feature values.
            """)
else:
    st.markdown("---")
    st.markdown("### 👈 Adjust parameters in the sidebar and click 'Run Clustering' to start analysis!")
