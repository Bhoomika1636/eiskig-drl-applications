from __future__ import annotations
import glob
import os
import sys
# Add the directory containing controllerFunctions.py to the Python module search path
sys.path.append('experiments_hr')
import common.helpers as helpers
def main() -> None:
    targetfolderpath = 'experiments_hr/ReducedBoschSystem/results'
    targetsubfolders = ["ppo_red_2","ppo_red_3"]
    helpers.purgeWasteFiles(targetfolderpath,targetsubfolders)
    print("All specified files successfully removed!")

if __name__ == "__main__":
    main()