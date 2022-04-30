#ifndef ACTIVE_PANOPTIC_MAPPING_PLANNER_PANOPTIC_MAP_H_
#define ACTIVE_PANOPTIC_MAPPING_PLANNER_PANOPTIC_MAP_H_

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <active_3d_planning_core/data/trajectory.h>
#include <active_3d_planning_core/map/occupancy_map.h>
#include <panoptic_mapping/map/submap.h>
#include <panoptic_mapping/tools/planning_interface.h>
#include <panoptic_mapping_ros/panoptic_mapper.h>

namespace active_3d_planning {
namespace map {

// Map that implements access to a panoptic single TSDF map.
class PanopticMap : public OccupancyMap {
 public:
  explicit PanopticMap(PlannerI& planner);  // NOLINT
  virtual ~PanopticMap() = default;

  void setupFromParamMap(Module::ParamMap* param_map) override;

  // Access to the original map setMap needs to be called before the map can be
  // used.
  void setMap(std::shared_ptr<const panoptic_mapping::PanopticMapper> map);

  const panoptic_mapping::SubmapCollection& getMap() const {
    return map_->getSubmapCollection();
  }

  // Implement map interfaces.
  bool isTraversable(const Eigen::Vector3d& position,
                     const Eigen::Quaterniond& orientation =
                         Eigen::Quaterniond(1, 0, 0, 0)) override;

  bool isTraversablePath(const EigenTrajectoryPointVector& trajectory) override;
  bool isObserved(const Eigen::Vector3d& point) override;
  unsigned char getVoxelState(const Eigen::Vector3d& point) override;
  double getVoxelSize() override;
  bool getVoxelCenter(Eigen::Vector3d* center,
                      const Eigen::Vector3d& point) override;

 private:
  static ModuleFactoryRegistry::Registration<PanopticMap> registration_;
  std::shared_ptr<const panoptic_mapping::PanopticMapper> map_;

  // methods
  bool checkIsSetup();

  // Params
  float p_check_collision_distance_;  // m
  float p_voxel_size_;  // m, virtual grid used for view computation.

  // Variables
  bool is_setup_ = false;
  const panoptic_mapping::Submap* active_submap_ = nullptr;
  const panoptic_mapping::Submap* past_submap_ = nullptr;

  // Cached data.
  std::vector<Eigen::Vector3d> c_collision_offsets_;
};

}  // namespace map
}  // namespace active_3d_planning

#endif  // ACTIVE_CHANGE_DETECTION_PLANNER_PANOPTIC_MAP_H_
