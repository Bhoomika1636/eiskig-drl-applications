import sys
from pathlib import Path

# load local package
current_file_path = Path(__file__)
root_path = current_file_path.parent.parent
sys.path.append(str(root_path))
from common.ksf import kmeans_clustering_filter

action_selection = [
    "bSetStatusOn_HeatExchanger1",
    "bSetStatusOn_CHP1",
    "bSetStatusOn_CHP2",
    "bSetStatusOn_CondensingBoiler",
    "bSetStatusOn_VSIStorage",
    "bLoading_VSISystem",
    "bSetStatusOn_HVFASystem_HNLT",
    "bLoading_HVFASystem_HNLT",
    "bSetStatusOn_eChiller",
    "bSetStatusOn_HVFASystem_CN",
    "bLoading_HVFASystem_CN",
    "bSetStatusOn_OuterCapillaryTubeMats",
    "bSetStatusOn_HeatPump",
]

kmeans_clustering_filter(
    file_path="experiments_hr/supplysystem_ETA/results/Three_days_2017_ETA/3_days_000-01_episode.csv",
    separator=";",
    decimal=".",
    column_selection=action_selection,
    number_of_buckets=360,
    min_clusters=1,
    max_clusters=12,
)
