## active_panoptic_mapping_utils

This package includes a collection of utilities and script to prepare the data for and evaluate both mapping and planning experiments.

## ScanNet label integration experiments

1. Download the ScanNetV2 dataset according to [the official installation instructions](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation). You can limit to the scans listed in [the experiments config file](./config/experiments/mapping/scannet.yaml).

1. Extract the data in the scans using the [extract_scans_data.py](./scripts/extracts_scans_data.sh) script:

    ```
    bash scripts/extract_scans_data.sh <SCANNET_SCANS_DIR>
    ```

1. Generate panoptic labels using Mask2Former:
    ```
    bash scripts/predict_scannet_panoptic_labels.sh <SCANNET_SCANS_DIR>
    ```

1. In `scannet.yaml`, set `out_dir` to a directory on your system where experiments results should be saved and `data_dir` to the directory containing the ScanNet scans to evaluate on.

1. Run the experiments:
    ```
    roslaunch active_panoptic_mapping_utils run_mapping_experiments.launch
    ```
1. Create ground truth vertex panoptic labels for evaluation:
    ```
    python3 scripts/create_scannet_ground_truth_vertex_labels.py <SCANNET_SCANS_DIR>
    ```
1. Evalaute the experiments:
    ```
    evaluate_scannet_mapping_experiments.sh <EXPERIMENTS_DIR> <SCANNET_SCANS_DIR>
    ```
1. (Optional) Visualize the results i.e. create meshes with the same geometry as the ground truth ones but each vertex colored according to the predicted panoptic label:
    ```
    visualize_mapping_experiments_results.sh <EXPERIMENTS_DIR> <SCANNET_SCANS_DIR>
    ```

## Replica planning experiments

1. Make sure the Replica Dataset was downloaded correctly.

1. Run the experiments:
    ```
    bash scripts/run_planning_experiments.sh <EXPERIMENTS_DIR>
    ```
   This script will run each planner configuration 5 times on the Replica FRL-Apartment-0 scene, each time for 30 minutes.

1. Generate the ground truth vertex panoptic labels for Replica:
    ```
    python3 scripts/create_replica_ground_truth_vertex_labels.py <REPLICA_DIR>
    ```
1. Evaluate the experiments:
    ```
    bash evaluate_replica_planning_experiments.sh <PLANNING_EXPERIMENTS_DIR> <REPLICA_DIR>
    ```