def visualize_metrics(metrics):
    """
    Visualize clustering quality metrics
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing quality metrics for each k
    """
    # Get output directory
    output_dir = 'kmeans_results'
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')

# For reproducibility
np.random.seed(42)

# Create output directory for saved files
def create_output_directory():
    """Create output directory for saved files if it doesn't exist"""
    output_dir = 'kmeans_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

# Load the dataset
# In a real scenario, you would load the actual data files
# For this example, we'll generate synthetic data based on the features information
# You would replace this with loading your actual dataset
def load_synthetic_har_data():
    """
    Create synthetic data that mimics the HAR dataset structure
    In a real scenario, you would load the actual files
    """
    # Number of samples
    n_samples = 1000
    
    # Number of features (simplified - we'll use a subset of the 561 features)
    n_features = 50
    
    # Create synthetic features with some structure
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels (6 activities as seen in activity_labels.txt)
    y = np.random.randint(1, 7, size=n_samples)
    
    # Create feature names based on the first n_features from features.txt
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    
    # Create activity names based on activity_labels.txt
    activity_map = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING"
    }
    
    # Map numeric labels to activity names
    y_names = [activity_map[label] for label in y]
    
    return X, y, y_names, feature_names, activity_map

# Load the synthetic data
X, true_labels, activity_names, feature_names, activity_map = load_synthetic_har_data()

print(f"Dataset shape: {X.shape}")
print(f"Number of activities: {len(np.unique(true_labels))}")
print(f"Activities: {', '.join(list(activity_map.values()))}")

# Preprocess the data
def preprocess_data(X):
    """
    Preprocess the data by scaling and optionally reducing dimensionality
    """
    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

# Experimental protocol for k-means
def run_kmeans_experiment(X, k_values, n_init=10, max_iter=300, random_state=42):
    """
    Run k-means experiments with different values of k
    
    Parameters:
    -----------
    X : array-like
        Input data
    k_values : list
        List of k values to try
    n_init : int
        Number of initializations
    max_iter : int
        Maximum number of iterations
    random_state : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary containing k-means models and labels for each k
    """
    results = {}
    
    for k in k_values:
        print(f"Running k-means with k={k}...")
        
        # Initialize and fit k-means
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',  # Using k-means++ initialization
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        
        # Fit the model
        kmeans.fit(X)
        
        # Get cluster assignments
        labels = kmeans.labels_
        
        # Store results
        results[k] = {
            'model': kmeans,
            'labels': labels
        }
    
    return results

# Calculate clustering quality metrics
def calculate_clustering_quality(X, kmeans_results):
    """
    Calculate various clustering quality metrics
    
    Parameters:
    -----------
    X : array-like
        Input data
    kmeans_results : dict
        Dictionary containing k-means results
        
    Returns:
    --------
    dict
        Dictionary containing quality metrics for each k
    """
    metrics = {}
    
    for k, result in kmeans_results.items():
        labels = result['labels']
        model = result['model']
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = model.inertia_
        
        # Calculate silhouette score
        silhouette = silhouette_score(X, labels) if k > 1 else 0
        
        # Calculate Calinski-Harabasz Index (Variance Ratio Criterion)
        calinski_harabasz = calinski_harabasz_score(X, labels) if k > 1 else 0
        
        # Calculate Davies-Bouldin Index
        davies_bouldin = davies_bouldin_score(X, labels) if k > 1 else float('inf')
        
        metrics[k] = {
            'inertia': inertia,
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz,
            'davies_bouldin': davies_bouldin
        }
        
        print(f"k={k}: Inertia={inertia:.2f}, Silhouette={silhouette:.4f}, "
              f"CH={calinski_harabasz:.2f}, DB={davies_bouldin:.4f}")
    
    return metrics

# Evaluate clustering against true labels
def evaluate_clustering(true_labels, kmeans_results, activity_map):
    """
    Evaluate clustering results against true labels
    
    Parameters:
    -----------
    true_labels : array-like
        True activity labels
    kmeans_results : dict
        Dictionary containing k-means results
    activity_map : dict
        Mapping from activity IDs to names
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    evaluation = {}
    
    for k, result in kmeans_results.items():
        cluster_labels = result['labels']
        
        # Create contingency table (cluster vs true labels)
        contingency = pd.crosstab(
            pd.Series(cluster_labels, name='Cluster'),
            pd.Series([activity_map[l] for l in true_labels], name='Activity')
        )
        
        # Calculate cluster homogeneity: what percentage of each cluster belongs to the same true class
        cluster_homogeneity = {}
        for cluster_id in range(k):
            if cluster_id in contingency.index:
                cluster_counts = contingency.loc[cluster_id]
                total = cluster_counts.sum()
                max_class = cluster_counts.idxmax()
                max_count = cluster_counts.max()
                homogeneity = max_count / total if total > 0 else 0
                cluster_homogeneity[cluster_id] = {
                    'dominant_activity': max_class,
                    'homogeneity': homogeneity
                }
        
        evaluation[k] = {
            'contingency': contingency,
            'cluster_homogeneity': cluster_homogeneity
        }
    
    return evaluation

# Visualize clusters in reduced dimensions
def visualize_clusters(X, kmeans_results, true_labels, activity_map):
    """
    Visualize clusters in 2D using PCA
    
    Parameters:
    -----------
    X : array-like
        Input data
    kmeans_results : dict
        Dictionary containing k-means results
    true_labels : array-like
        True activity labels
    activity_map : dict
        Mapping from activity IDs to names
    """
    # Get output directory
    output_dir = 'kmeans_results'
    
    # Apply PCA to reduce to 2 dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Set up the figure
    for k, result in kmeans_results.items():
        if k > 10:  # Skip very large k for clarity
            continue
            
        cluster_labels = result['labels']
        
        # Create a figure with 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Scatter plot colored by cluster assignment
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', 
                               alpha=0.7, s=30)
        ax1.set_title(f'K-means Clustering (k={k})')
        ax1.set_xlabel('PCA Component 1')
        ax1.set_ylabel('PCA Component 2')
        legend1 = ax1.legend(*scatter1.legend_elements(), title="Clusters")
        ax1.add_artist(legend1)
        
        # Plot 2: Scatter plot colored by true activity
        activity_numeric = np.array([list(activity_map.keys())[list(activity_map.values()).index(name)] 
                                     for name in [activity_map[l] for l in true_labels]])
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=activity_numeric, cmap='tab10', 
                               alpha=0.7, s=30)
        ax2.set_title('True Activities')
        ax2.set_xlabel('PCA Component 1')
        ax2.set_ylabel('PCA Component 2')
        legend2 = ax2.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
                                                  markerfacecolor=plt.cm.tab10(i/6), markersize=10) 
                                      for i in range(6)],
                             labels=list(activity_map.values()),
                             title="Activities")
        ax2.add_artist(legend2)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'kmeans_clusters_k{k}.png'), dpi=300, bbox_inches='tight')
        print(f"Saved cluster visualization for k={k} to {output_dir}/kmeans_clusters_k{k}.png")
        
        plt.show()

# Save quality metrics to CSV
def save_metrics_to_csv(metrics):
    """
    Save clustering quality metrics to CSV file
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing quality metrics for each k
    """
    # Convert metrics dictionary to DataFrame
    metrics_df = pd.DataFrame({
        'k': list(metrics.keys()),
        'inertia': [metrics[k]['inertia'] for k in metrics],
        'silhouette': [metrics[k]['silhouette'] for k in metrics],
        'calinski_harabasz': [metrics[k]['calinski_harabasz'] for k in metrics],
        'davies_bouldin': [metrics[k]['davies_bouldin'] for k in metrics]
    })
    
    # Sort by k
    metrics_df = metrics_df.sort_values('k')
    
    # Save to CSV
    metrics_df.to_csv('kmeans_quality_metrics.csv', index=False)
    print("Saved quality metrics to kmeans_quality_metrics.csv")

# Visualize elbow plot and other metrics
def visualize_metrics(metrics):
    """
    Visualize clustering quality metrics
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing quality metrics for each k
    """
    k_values = sorted(metrics.keys())
    inertia = [metrics[k]['inertia'] for k in k_values]
    silhouette = [metrics[k]['silhouette'] for k in k_values]
    calinski_harabasz = [metrics[k]['calinski_harabasz'] for k in k_values]
    davies_bouldin = [metrics[k]['davies_bouldin'] for k in k_values]
    
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot inertia (Elbow plot)
    axs[0, 0].plot(k_values, inertia, 'bo-')
    axs[0, 0].set_title('Elbow Plot (Inertia)')
    axs[0, 0].set_xlabel('Number of clusters (k)')
    axs[0, 0].set_ylabel('Inertia')
    axs[0, 0].grid(True)
    
    # Plot silhouette score
    axs[0, 1].plot(k_values, silhouette, 'go-')
    axs[0, 1].set_title('Silhouette Score')
    axs[0, 1].set_xlabel('Number of clusters (k)')
    axs[0, 1].set_ylabel('Silhouette Score')
    axs[0, 1].grid(True)
    
    # Plot Calinski-Harabasz Index
    axs[1, 0].plot(k_values, calinski_harabasz, 'ro-')
    axs[1, 0].set_title('Calinski-Harabasz Index')
    axs[1, 0].set_xlabel('Number of clusters (k)')
    axs[1, 0].set_ylabel('Calinski-Harabasz Index')
    axs[1, 0].grid(True)
    
    # Plot Davies-Bouldin Index
    axs[1, 1].plot(k_values, davies_bouldin, 'mo-')
    axs[1, 1].set_title('Davies-Bouldin Index')
    axs[1, 1].set_xlabel('Number of clusters (k)')
    axs[1, 1].set_ylabel('Davies-Bouldin Index')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save the metrics figure
    plt.savefig('kmeans_quality_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved quality metrics visualization to kmeans_quality_metrics.png")
    
    plt.show()

# Main execution code
def main():
    print("K-means Clustering Analysis on HAR Dataset")
    print("==========================================")
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Preprocess the data
    print("\nPreprocessing data...")
    X_processed = preprocess_data(X)
    
    # Define k values to try
    # We include k=6 to match the number of activities in the dataset
    k_values = [2, 3, 4, 5, 6, 8, 10, 12]
    
    # Run k-means experiment
    print("\nRunning k-means experiments with different k values...")
    kmeans_results = run_kmeans_experiment(X_processed, k_values)
    
    # Calculate clustering quality metrics
    print("\nCalculating clustering quality metrics...")
    quality_metrics = calculate_clustering_quality(X_processed, kmeans_results)
    
    # Save metrics to CSV
    save_metrics_to_csv(quality_metrics)
    
    # Evaluate clustering against true labels
    print("\nEvaluating clustering against true activity labels...")
    evaluation = evaluate_clustering(true_labels, kmeans_results, activity_map)
    
    # Display detailed evaluation for k=6 (matching number of activities)
    if 6 in evaluation:
        print("\nDetailed evaluation for k=6 (matching number of activities):")
        contingency_table = evaluation[6]['contingency']
        print(contingency_table)
        
        # Save contingency table to CSV
        contingency_table.to_csv(os.path.join(output_dir, 'contingency_table_k6.csv'))
        print(f"Saved contingency table for k=6 to {output_dir}/contingency_table_k6.csv")
        
        print("\nCluster homogeneity:")
        for cluster_id, details in evaluation[6]['cluster_homogeneity'].items():
            print(f"Cluster {cluster_id}: {details['dominant_activity']} "
                  f"({details['homogeneity']:.2%} homogeneity)")
    
    # Visualize results
    print("\nVisualizing clusters...")
    visualize_clusters(X_processed, kmeans_results, true_labels, activity_map)
    
    # Visualize metrics
    print("\nVisualizing quality metrics...")
    visualize_metrics(quality_metrics)
    
    # Conclusion
    print("\nConclusion:")
    
    # Find optimal k based on metrics
    best_k_silhouette = max(quality_metrics, key=lambda k: quality_metrics[k]['silhouette'])
    best_k_calinski = max(quality_metrics, key=lambda k: quality_metrics[k]['calinski_harabasz'])
    best_k_davies = min(quality_metrics, key=lambda k: quality_metrics[k]['davies_bouldin'])
    
    print(f"Best k according to Silhouette Score: {best_k_silhouette}")
    print(f"Best k according to Calinski-Harabasz Index: {best_k_calinski}")
    print(f"Best k according to Davies-Bouldin Index: {best_k_davies}")
    
    # Compare with ground truth (6 activities)
    print(f"\nGround truth has 6 activities. Our metrics suggest optimal k values of "
          f"{best_k_silhouette}, {best_k_calinski}, and {best_k_davies}.")
    
    # Write results to a summary file
    with open(os.path.join(output_dir, 'kmeans_summary.txt'), 'w') as f:
        f.write("K-means Clustering Analysis on HAR Dataset\n")
        f.write("==========================================\n\n")
        f.write(f"Best k according to Silhouette Score: {best_k_silhouette}\n")
        f.write(f"Best k according to Calinski-Harabasz Index: {best_k_calinski}\n")
        f.write(f"Best k according to Davies-Bouldin Index: {best_k_davies}\n\n")
        f.write(f"Ground truth has 6 activities. Our metrics suggest optimal k values of "
                f"{best_k_silhouette}, {best_k_calinski}, and {best_k_davies}.\n")
    
    print(f"\nSaved summary results to {output_dir}/kmeans_summary.txt")

# Run the main function
if __name__ == "__main__":
    main()