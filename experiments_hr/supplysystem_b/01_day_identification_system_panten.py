import sys
from pathlib import Path
import json

# load local package
current_file_path = Path(__file__)
root_path = current_file_path.parent.parent
sys.path.append(str(root_path))
from common.ksf import kmeans_clustering_filter


# set correct directory and saving path name
directory = current_file_path.parent / "results" / "day_identification"

# function to store results
def save_json(run_name, cluster_result):
    file_path_json = directory / str(run_name + ".json")
    if not directory.exists():
        directory.mkdir(parents=True)
    with open(file_path_json, "w") as json_file:
        json.dump(cluster_result, json_file)


action_selection = [
    "u_combinedheatpower",
    "u_condensingboiler",
    "u_immersionheater",
    "u_heatpump",
    "u_coolingtower",
    "u_compressionchiller",
]

cluster_dict = kmeans_clustering_filter(
    file_path="experiments_hr/supplysystem_b/results/2017_360_days_P2/2017_360_days_P2_000-01_episode.csv",
    separator=";",
    decimal=".",
    column_selection=action_selection,
    number_of_buckets=360,
    min_clusters=1,
    max_clusters=12,
)

print("[INFO] Cluster centers and elements per cluster:", cluster_dict)
save_json(run_name="clustering_result", cluster_result=cluster_dict)