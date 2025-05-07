import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_breast_cancer, load_iris, load_wine


# -----------------------------------------------
# Helper Functions
# -----------------------------------------------

# Preprocess data: handle missing values and categorical variables
def preprocess_data(X):
    # Drop rows with any missing values
    X_clean = X.dropna()

    # Identify categorical columns
    categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns

    # Convert categorical columns using one-hot encoding
    if len(categorical_cols) > 0:
        X_clean = pd.get_dummies(X_clean, columns=categorical_cols, drop_first=True)

    return X_clean


# Loading sample datasets from sklearn
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

# Standardize features 
def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Run the KMeans ML model
def run_kmeans(X, k):
    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(X)
    return kmeans, clusters

# Run the hierarchical clustering model with Ward linkage
def run_hierarchical(X, k, linkage_method='ward'):
    hierarchical = AgglomerativeClustering(n_clusters=k, linkage='ward')
    clusters = hierarchical.fit_predict(X)
    return hierarchical, clusters

# Compute linkage matrix for dendrogram using Ward method
def compute_linkage_matrix(X, method='ward'):
    Z = linkage(X, method='ward')
    return Z

# Plot dendrogram for hierarchical clustering
def plot_dendrogram(Z, labels=None, max_d=None):
    plt.figure(figsize=(10, 7))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    
    # If we have too many samples, truncate the dendrogram
    if Z.shape[0] > 50:  # If more than 50 samples
        dendrogram(
            Z,
            truncate_mode='lastp',
            p=30,  # Show only the last 30 merges
            leaf_rotation=90.,
            leaf_font_size=8.,
            labels=labels
        )
    else:
        dendrogram(
            Z,
            leaf_rotation=90.,
            leaf_font_size=8.,
            labels=labels
        )
    
    if max_d:
        plt.axhline(y=max_d, c='k', linestyle='--')
    
    fig = plt.gcf()
    return fig

# Compute data for elbow method and silhouette score
def compute_metrics(X, method, max_k=10):
    wcss = []
    silhouette_scores = []
    ks = range(2, max_k + 1)
    
    for i in ks:
        if method == 'kmeans':
            km = KMeans(n_clusters=i)
            km.fit(X)
            wcss.append(km.inertia_)
            labels = km.labels_
        else:  # hierarchical
            hc = AgglomerativeClustering(n_clusters=i)
            labels = hc.fit_predict(X)
            wcss.append(None)  # Hierarchical doesn't have inertia
        
        silhouette_scores.append(silhouette_score(X, labels))
    
    return ks, wcss, silhouette_scores

# Apply PCA
def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_

# Plot clusters using PCA
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
    plt.title('Clustering: 2D PCA Projection')
    plt.legend(loc='best')
    plt.grid(True)
    
    fig = plt.gcf()
    return fig

# Plot optimal k analysis (elbow method and silhouette scores)
def plot_optimal_k(ks, wcss, silhouette_scores, method):
    plt.figure(figsize=(12, 5))

    # If KMeans, include Elbow Method
    if method == 'kmeans' and wcss[0] is not None:
        plt.subplot(1, 2, 1)
        plt.plot(ks, wcss, 'o-', color='blue')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares')
        plt.title('Elbow Method')
        plt.grid(True)
        
        # Silhouette Score
        plt.subplot(1, 2, 2)

    # Just silhouette if hierarchical or if wcss is None
    else:
        plt.subplot(1, 1, 1)
    
    plt.plot(ks, silhouette_scores, 'o-', color='green')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    plt.grid(True)

    plt.tight_layout()
    fig = plt.gcf()
    return fig


# -----------------------------------------------
# Main App Layout
# -----------------------------------------------

# Set page configuration
st.set_page_config(
    page_title="Advanced Clustering Explorer",
    page_icon="📊",
    layout="wide"
)

# Title and introduction
st.title("📊 Advanced Clustering Explorer")
st.markdown("""
This app allows you to explore both KMeans and Hierarchical clustering on various datasets.
Upload your own dataset or use one of the sample datasets to discover natural groups in your data.
""")

# Sidebar for dataset selection and parameters
st.sidebar.header("Settings")

# Dataset selection
dataset_option = st.sidebar.radio(
    "Choose a dataset:",
    ["Breast Cancer", "Iris", "Wine", "Upload Your Own"]
)

# Check if the user selected the "Upload Your Own" option from the sidebar
if dataset_option == "Upload Your Own":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Ask the user to select a target column from the DataFrame (optional)
        target_col = st.sidebar.selectbox(
            "Select target column (optional):",
            ["None"] + df.columns.tolist()
        )
        
        # If a target column is selected
        if target_col != "None":
             # Use all columns except the target as features
            feature_cols = [col for col in df.columns if col != target_col]
            X = df[feature_cols]
            y = df[target_col]
        # If no target column is selected, treat all columns as features
        else:
            X = df
            y = None
        # Preprocess data
        X = preprocess_data(X)

    else:
        st.markdown("### 👈 Please upload a CSV file or select a sample dataset.")
        st.stop()
else:
    X, y = load_sample_dataset(dataset_option)

    # Make target column selectable for sample datasets
    target_col = st.sidebar.selectbox(
        "Select target column (optional):",
        ["None", "target"] if y is not None else ["None"]
    )

    if target_col != "None":
        y = y
    else:
        y = None

# Display dataset information
st.subheader("Dataset Overview")
st.write(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

st.markdown("""
- **Samples**: rows = individual observations (like one flower or one patient).
- **Features**: columns = measurements or characteristics (like petal width or alcohol content).
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
        target_info = "The target variable represents different wine cultivators (0, 1, 2)."
    elif dataset_option == "Upload Your Own":
        target_info = f"Using '{target_col}' as the target variable."
    
    st.write(target_info)

    st.markdown("*Note: This is an unsupervised model so it does not use the target to make predictions, the target is only to compare results to afterwards*")

# Clustering Method Selection
st.sidebar.header("Clustering Method")
clustering_method = st.sidebar.radio(
    "Select clustering algorithm:",
    ["KMeans", "Hierarchical"]
)

# Common parameters
st.sidebar.header("Clustering Parameters")
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10)
scaling = st.sidebar.checkbox("Scale features", value=True)

# Display explanation of Ward linkage method for hierarchical clustering
if clustering_method == "Hierarchical":
    st.sidebar.markdown("""
    **Ward Linkage Method:**
    - Minimizes the sum of squared differences within clusters (similar to KMeans)
    - (Solid for most applications as it's robust to noise)
    """)

# Run clustering when requested
if st.sidebar.button(f"Run {clustering_method} Clustering"):
    # Scale features if selected and notify user
    if scaling:
        X_processed = scale_features(X)
        st.success("Data scaled")
    else:
        X_processed = X.values
        st.warning(f"Using unscaled data. {clustering_method} works best with scaled features for most datasets.")

    st.markdown("*Scaling normalizes features to a similar range (mean=0, std=1). This prevents features with larger values from dominating the distance calculations.*")
    
    # Run selected clustering algorithm
    if clustering_method == "KMeans":
        model, cluster_labels = run_kmeans(X_processed, k)
    else:  # Hierarchical with Ward linkage
        model, cluster_labels = run_hierarchical(X_processed, k, 'ward')

    # Add cluster labels to the original data
    clustered_data = X.copy()
    clustered_data['Cluster'] = cluster_labels

    # Only add True_Label if a target is actually selected
    if y is not None:
        clustered_data['True_Label'] = y.values if hasattr(y, 'values') else y
        # Move these columns to the front with True_Label
        clustered_data = clustered_data[['Cluster', 'True_Label'] + [col for col in clustered_data.columns if col not in ['Cluster', 'True_Label']]]
    else:
        # Only move Cluster to the front if no True_Label
        clustered_data = clustered_data[['Cluster'] + [col for col in clustered_data.columns if col != 'Cluster']]

    # Create tabs for different analysis
    if clustering_method == "KMeans":
        tab1, tab2, tab3 = st.tabs(["Cluster Visualization", "Optimal K Analysis", "Cluster Data"])
    else:  # Hierarchical
        tab1, tab2, tab3, tab4 = st.tabs(["Cluster Visualization", "Dendrogram", "Optimal K Analysis", "Cluster Data"])
    
    with tab1:
        st.subheader("2D Visualization of Clusters")
        
        # Apply PCA 
        X_pca, explained_variance = apply_pca(X_processed)
        st.write(f"Explained variance: {sum(explained_variance):.2%}")
        
        # Plot clusters
        fig = plot_clusters(X_pca, cluster_labels)
        st.pyplot(fig)

        with st.expander("PCA Explained"):
            st.markdown("""
                To help visualize your dataset, we used a technique called **PCA (Principal Component Analysis)**. Most datasets have many features (columns), which makes them hard to plot directly.
                - PCA looks for the directions (combinations of features) where the data varies the most.
                - It then **rotates and compresses** the data along those directions to create a 2D view.
            """)

        if clustering_method == "KMeans":
            st.markdown("""
                Each dot in the plot represents a data point, and the color shows which **cluster** it was assigned to by the KMeans algorithm.
                
                **How the KMeans algorithm works:**
                - It first randomly picks **k** points as cluster centers (called *centroids*).
                - Each data point is assigned to the nearest centroid based on distance.
                - The centroids are then updated to the average position of the points assigned to them.
                - This repeats until the cluster assignments stop changing significantly (called *convergence*).
            """)
        else:  # Hierarchical
            st.markdown("""
                Each dot in the plot represents a data point, and the color shows which **cluster** it was assigned to by the Hierarchical clustering algorithm.
                
                **How Hierarchical Clustering works:**
                - It starts by treating each data point as its own cluster.
                - Then, it repeatedly merges the two most similar clusters.
                - The way it measures similarity between clusters depends on the linkage method.
                - The process continues until all points belong to a single cluster.
                - We then "cut" the resulting tree at a point that gives us **k** clusters.
            """)
    
    # Only show dendrogram tab for hierarchical clustering
    if clustering_method == "Hierarchical":
        with tab2:
            st.subheader("Hierarchical Clustering Dendrogram")
            
            # Compute linkage matrix for dendrogram using Ward method
            Z = compute_linkage_matrix(X_processed, method='ward')
            
            # Plot dendrogram
            fig = plot_dendrogram(Z)
            st.pyplot(fig)

            # Explaining Linkage method
            with st.expander('Ward Linkage Metho details'):
                st.markdown("""
                    **Ward Linkage Method:**
                    - Minimizes the sum of squared differences within clusters (similar to KMeans)
                    - (Solid for most applications as it's robust to noise)
                            """)
                
            st.markdown("""
                **How to read a dendrogram:**
                - The dendrogram shows the hierarchical relationship between clusters.
                - The y-axis represents the distance or dissimilarity between clusters.
                    - Each vertical line corresponds to a merge between two clusters.
                    - The height of the vertical line indicates the dissimilarity between the merged clusters.
                - The x-axis represents individual data points or clusters.
                - To get **k** clusters, imagine drawing a horizontal line across the dendrogram and count the number of vertical lines it intersects!
            """)

    # Tab for optimal k analysis
    with tab2 if clustering_method == "KMeans" else tab3:
        st.subheader("Finding the Optimal Number of Clusters")
        
        # Compute metrics for optimal k
        ks, wcss, silhouette_scores = compute_metrics(X_processed, 
                                                     'kmeans' if clustering_method == "KMeans" else 'hierarchical')
        
        # Plot optimal k analysis
        fig = plot_optimal_k(ks, wcss, silhouette_scores, 
                           'kmeans' if clustering_method == "KMeans" else 'hierarchical')
        st.pyplot(fig)

        # Add explanations
        if clustering_method == "KMeans":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Elbow Method**
                
                This plot shows how the Within-Cluster Sum of Squares (WCSS) decreases as we add more clusters.
                
                - **What to look for:** The "elbow" point where adding more clusters gives diminishing returns
                - **How it works:** WCSS measures how compact clusters are (lower is better)
                - **Best K value:** Where the curve bends like an elbow
                """)
        
        # This explanation is shown for both methods
        with col2 if clustering_method == "KMeans" else st.container():
            st.markdown("""
            **Silhouette Score**
            
            This plot shows how well-separated the clusters are at different K values.
            
            - **What to look for:** The highest point on the curve
            - **How it works:** Scores range from -1 to 1, where higher values mean better-defined clusters
            - **Best K value:** The K with the highest silhouette score
            """)
    
    # Tab for clustered data
    with tab3 if clustering_method == "KMeans" else tab4:
        st.subheader("Clustered Data Results")
        
        # Calculate and display cluster sizes
        st.write("### Cluster Sizes")
        cluster_counts = clustered_data['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)
        
        # Display Dataframe w cluster and target labels
        st.write("### Data with Cluster Assignments")
        st.dataframe(clustered_data)

        # Only show this explanation if there's a target variable
        if y is not None:
            st.info(f"""
                **ℹ️ Note:** When using a sample dataset like *Breast Cancer*, the original class labels (e.g., *malignant* vs. *benign*) are available and shown as `True_Label`.
                This allows you to compare the clustering results (`Cluster`) to the actual known classes (`True_Label`).
                
                Keep in mind that the **{clustering_method} model is an unsupervised algorithm** — it doesn't use or know these labels.
            """)
else:
    st.markdown("---")
    st.markdown(f"### 👈 Adjust parameters in the sidebar and click 'Run {clustering_method} Clustering' to start analysis!")
