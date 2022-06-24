## active_panoptic_mapping_core

Core library of the `active_panoptic_mapping` meta-package. It provides the following:
- A Single TSDF tracker for `panoptic_mapping` which can track panoptic labels.
- Trajectory evaluators for `mav_active_3d_planning` which use the voxel label uncertainty in a `panoptic_mapping` map to compute the information gain.
- Utilities for online inference with Mask2Former.

### Note

Unfortunately, some of the functionality required for the mapper to work had to be implemented directly as part of the `panoptic_mapping` package. Therefore, we require a specific version of that package which is automatically downloaded during setup.