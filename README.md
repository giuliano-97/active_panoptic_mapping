# Active panoptic mapping


## Setup

### Dependencies

This project uses [Robostack](https://robostack.github.io/) so only a minimal `conda` installation such as [`mambaforge`](https://github.com/conda-forge/miniforge) is required to get started. 

### Installation


1. Create a directory tree for catkin workspace:

    ```
    mkdir -p catkin_ws/src 
    cd catkin_ws/src
    ```
1. Clone this repository in the src directory:

    ```
    git clone git@github.com:giuliano-97/active_panoptic_mapping.git
    ```

1. Make sure `mamba` is install in your `base` environment (not needed when using `mambaforge`), then create the conda environment by running:

    ```
    mamba env create -f environment.yml
    ```
1. Activate the create conda environment:
    ```
    conda activate active_panoptic_mapping_env
    ```
1. Initialize the catkin workspace:
    ```
    cd .. && catkin init
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
    catkin config --merge-devel
    cd src
    ```
1. Install package dependencies using ros install:
    - If you created a new workspace:
    ```
    wstool init . /active_panoptic_mapping/active_panoptic_mapping_ssh.rosinstall
    wstool update
    ```
    - If you use an existing workspace. Notice that some dependencies require specific branches that will be checked out.
    ```
    wstool merge -t . ./active_panoptic_mapping/active_panoptic_mapping.rosinstall
    wstool update
    ```
1. Apply some minor patches to the `eigen_catkin` and `opencv_catkin` packages to avoid compilation errors:
    ```
    bash active_panoptic_mapping/apply_patches.sh
    ```
1. Compile and source the ros environment:
    ```
    catkin build active_panoptic_mapping_utils
    source ../devel/setup.bash
    ```

### Replica Demo

1. Download the Replica dataset by following the download instructions [on the GitHub page](https://github.com/facebookresearch/Replica-Dataset#download-on-mac-os-and-linux). 
1. Launch the `active_panoptic_mapping_node`:
    ```
    roslaunch active_panoptic_mapping_ros run.launch datasets_dir:=$(dirname <PATH_TO_REPLICA_DIR>)

    ```