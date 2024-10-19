"""
K-Means Similarity Filter
=========================
This module provides functionality to perform k-means clustering on a dataset,
visualizing the results and the cosine similarity matrix of the data.

Functions:
- slice_dataframe_into_buckets: Slices the DataFrame into specified number of buckets.
- calculate_fractions: Calculates the fraction of 'on' (1.0) values in each bucket.
- cosine_similarity: Computes the cosine similarity between two vectors.
- calculate_all_cosine_similarities: Calculates the cosine similarity matrix for all buckets.
- bucket_fractions_to_array: Converts bucket fractions to a NumPy array.
- perform_kmeans_clustering: Performs k-means clustering on the data.
- kmeans_clustering_filter: Main function to execute the clustering and plot results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans


def slice_dataframe_into_buckets(df, number_of_buckets):
    """
    Slices a DataFrame into a specified number of buckets based on total rows.

    Parameters:
    df (pd.DataFrame): The DataFrame to slice.
    number_of_buckets (int): The number of buckets to divide the DataFrame into.

    Returns:
    list: A list of DataFrames representing the sliced buckets.
    """
    total_rows = len(df)
    rows_per_bucket = total_rows // number_of_buckets
    buckets = []

    for i in range(number_of_buckets):
        start_index = i * rows_per_bucket
        end_index = total_rows if i == number_of_buckets - 1 else start_index + rows_per_bucket
        bucket = df.iloc[start_index:end_index]
        buckets.append(bucket)

    return buckets


def calculate_fractions(bucket):
    """
    Calculates the fraction of 1.0 values for each column in a bucket.

    Parameters:
    bucket (pd.DataFrame): The DataFrame bucket to calculate fractions for.

    Returns:
    dict: A dictionary of fractions for each column.
    """
    fractions = {column: (bucket[column] == 1.0).sum() / len(bucket[column]) for column in bucket.columns}
    return fractions


def cosine_similarity(bucket1, bucket2):
    """
    Computes the cosine similarity between two buckets (vectors).

    Parameters:
    bucket1 (np.array): The first bucket vector.
    bucket2 (np.array): The second bucket vector.

    Returns:
    float: The cosine similarity between the two buckets.
    """
    vector1 = np.array([val for val in bucket1.values()])
    vector2 = np.array([val for val in bucket2.values()])

    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0

    return dot_product / (norm_vector1 * norm_vector2)


def calculate_all_cosine_similarities(fractions_per_bucket):
    """
    Calculates cosine similarity between all pairs of buckets.

    Parameters:
    fractions_per_bucket (list): A list of fraction dictionaries for each bucket.

    Returns:
    np.array: A matrix of cosine similarities between buckets.
    """
    num_buckets = len(fractions_per_bucket)
    similarity_matrix = np.zeros((num_buckets, num_buckets))

    for i in range(num_buckets):
        for j in range(num_buckets):
            similarity = cosine_similarity(fractions_per_bucket[i], fractions_per_bucket[j])
            similarity_matrix[i, j] = similarity

    return similarity_matrix


def bucket_fractions_to_array(fractions_per_bucket):
    """
    Converts a list of fraction dictionaries to a NumPy array.

    Parameters:
    fractions_per_bucket (list): A list of fraction dictionaries for each bucket.

    Returns:
    np.array: An array of bucket fractions.
    """
    return np.array([list(bucket.values()) for bucket in fractions_per_bucket])


def perform_kmeans_clustering(fractions_array, number_of_clusters, random_state=17):
    """
    Performs k-means clustering on the given data.

    Parameters:
    fractions_array (np.array): Array of bucket fractions.
    number_of_clusters (int): The number of clusters to form.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: Cluster labels and cluster centers.
    """
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=random_state)
    kmeans.fit(fractions_array)
    return kmeans.labels_, kmeans.cluster_centers_


def kmeans_clustering_filter(
    file_path, separator, decimal, column_selection, number_of_buckets, min_clusters, max_clusters
):
    """
    Main function to execute k-means clustering and plot the results using the elbow method to determine the optimal number of clusters.

    Parameters:
    file_path (str): Path to the CSV file.
    separator (str): Separator used in the CSV file.
    decimal (str): Decimal point character used in the CSV file.
    column_selection (list): Columns to use from the CSV file.
    number_of_buckets (int): Number of buckets to divide the data into.
    min_clusters (int): Minimum number of clusters to try.
    max_clusters (int): Maximum number of clusters to try.

    Returns:
    None: The function creates plots and prints information directly.
    """
    df_actions = pd.read_csv(file_path, sep=separator, decimal=decimal, usecols=column_selection)

    buckets = slice_dataframe_into_buckets(df_actions, number_of_buckets)
    fractions_per_bucket = [calculate_fractions(bucket) for bucket in buckets]
    fractions_array = bucket_fractions_to_array(fractions_per_bucket)

    # Calculate cosine similarity matrix
    similarity_matrix = calculate_all_cosine_similarities(fractions_per_bucket)

    # Determine the optimal number of clusters using the elbow method
    wcss = []
    for i in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=17)
        kmeans.fit(fractions_array)
        wcss.append(kmeans.inertia_)

    # Plot the WCSS to visually inspect for the elbow
    plt.figure(figsize=(8, 4))
    plt.plot(range(min_clusters, max_clusters + 1), wcss, marker="o", color="black")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.xticks(range(min_clusters, max_clusters + 1))
    plt.grid(True)
    plt.show()

    # Manual inspection might be needed here to choose the optimal number of clusters
    # Replace the following line with your chosen number of clusters
    number_of_clusters = int(input("Enter the optimal number of clusters: "))

    # Perform k-means clustering with the chosen number of clusters
    labels, centers = perform_kmeans_clustering(fractions_array, number_of_clusters)

    # Plotting
    plt.figure(figsize=(12, 8))

    # First plot: Fractions with Cluster Color Background
    plt.subplot(2, 1, 1)
    cluster_colors = plt.cm.rainbow(np.linspace(0, 1, number_of_clusters))

    labels, _ = perform_kmeans_clustering(fractions_array, number_of_clusters)

    # Adjust the rectangles to start from half step back
    half_step = 0.5
    for i in range(number_of_buckets):
        cluster = labels[i]
        left_edge = i + half_step
        rect = Rectangle((left_edge, 0), 1, 1, color=cluster_colors[cluster], alpha=0.3, zorder=1)
        plt.gca().add_patch(rect)

    # Plot the "on" fractions
    for column in column_selection:
        plt.plot(
            range(1, number_of_buckets + 1),
            [fractions[column] for fractions in fractions_per_bucket],
            marker="o",
            markersize=3,
            linestyle="-",
            linewidth=0.75,
            label=column,
            zorder=2,
        )
    plt.title("Fraction of 1.0 values for each action over Buckets")
    plt.xlabel("Bucket Number")
    plt.ylabel("Fraction of 1.0")
    plt.xticks(np.arange(1, number_of_buckets + 1, max(1, number_of_buckets // 10)))
    plt.grid(True)
    plt.legend()

    # Plot cosine similarity matrix with adjusted ticks
    cos_sim_ax = plt.subplot(2, 2, 3)
    cmap = plt.cm.Greys
    norm = plt.Normalize(vmin=similarity_matrix.min(), vmax=similarity_matrix.max())
    img = plt.imshow(similarity_matrix, cmap=cmap, interpolation="nearest", norm=norm)
    plt.title("Cosine Similarity Matrix")
    plt.colorbar(img, ax=cos_sim_ax, label="Cosine Similarity")

    # Adjusting the number of tick marks based on the number of buckets
    tick_interval = max(1, number_of_buckets // 10)
    tick_marks = np.arange(0, number_of_buckets, tick_interval)
    plt.xticks(tick_marks, [str(i + 1) for i in tick_marks])
    plt.yticks(tick_marks, [str(i + 1) for i in tick_marks])
    plt.xlabel("Bucket Number")
    plt.ylabel("Bucket Number")
    plt.gca().invert_yaxis()

    # Fourth plot: Cluster Bar Chart with Cluster Colors and Value Labels
    plt.subplot(2, 2, 4)
    cluster_sizes = [len(np.where(labels == i)[0]) for i in range(number_of_clusters)]
    center_buckets = [np.argmin(np.linalg.norm(fractions_array - center, axis=1)) + 1 for center in centers]
    sorted_indices = np.argsort(center_buckets)
    sorted_cluster_sizes = np.array(cluster_sizes)[sorted_indices]
    sorted_center_buckets = np.array(center_buckets)[sorted_indices]

    # Create bars with correct colors
    bars = []
    for i in range(number_of_clusters):
        bar = plt.bar(i, sorted_cluster_sizes[i], color=cluster_colors[sorted_indices[i]], alpha=0.3)
        bars.append(bar)

    # Add value labels to each bar
    for bar in bars:
        for item in bar:
            yval = item.get_height()
            plt.text(item.get_x() + item.get_width() / 2, yval, int(yval), verticalalignment="bottom", ha="center")

    plt.title("Cluster Sizes and Center Buckets")
    plt.xlabel("Center Bucket of Cluster")
    plt.ylabel("Number of Buckets in Cluster")
    plt.xticks(range(number_of_clusters), sorted_center_buckets, rotation=45)

    plt.tight_layout()
    plt.show()

    # Creating the dictionary
    cluster_dict = {int(center): int(count) for center, count in zip(sorted_center_buckets, sorted_cluster_sizes)}

    return cluster_dict