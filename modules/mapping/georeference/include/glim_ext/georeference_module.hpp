#pragma once

#include <deque>
#include <atomic>
#include <thread>
#include <memory>
#include <string>
#include <unordered_set>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/format.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/concurrent_vector.hpp>
#include <glim/util/convert_to_string.hpp>
#include <glim_ext/util/config_ext.hpp>
#include <glim_ext/geodetic.hpp>

// ROS version-specific includes
#ifdef GLIM_ROS2
#include <glim/util/extension_module_ros2.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>

using ExtensionModuleBase = glim::ExtensionModuleROS2;
using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
using PoseWithCovarianceStampedConstPtr = geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using PoseStampedConstPtr = geometry_msgs::msg::PoseStamped::ConstSharedPtr;
using Odometry = nav_msgs::msg::Odometry;
using OdometryConstPtr = nav_msgs::msg::Odometry::ConstSharedPtr;
using NavSatFix = sensor_msgs::msg::NavSatFix;
using NavSatFixConstPtr = sensor_msgs::msg::NavSatFix::ConstSharedPtr;

// ROS2 timestamp conversion
template <typename Stamp>
double to_sec(const Stamp& stamp) {
  return stamp.sec + stamp.nanosec / 1e9;
}
#else
#include <glim/util/extension_module_ros.hpp>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>

using ExtensionModuleBase = glim::ExtensionModuleROS;
using PoseWithCovarianceStamped = geometry_msgs::PoseWithCovarianceStamped;
using PoseWithCovarianceStampedConstPtr = geometry_msgs::PoseWithCovarianceStampedConstPtr;
using PoseStamped = geometry_msgs::PoseStamped;
using PoseStampedConstPtr = geometry_msgs::PoseStampedConstPtr;
using Odometry = nav_msgs::Odometry;
using OdometryConstPtr = nav_msgs::OdometryConstPtr;
using NavSatFix = sensor_msgs::NavSatFix;
using NavSatFixConstPtr = sensor_msgs::NavSatFixConstPtr;

// ROS1 timestamp conversion
template <typename Stamp>
double to_sec(const Stamp& stamp) {
  return stamp.toSec();
}
#endif

// GTSAM includes
#include <spdlog/spdlog.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace glim {

// Symbol shorthand for GTSAM factors - using Symbol class directly
using gtsam::symbol_shorthand::X;  // For poses

// Constants for numerical stability and optimization
constexpr double MINIMUM_COVARIANCE_TRACE = 1e-9;
constexpr size_t MAX_PROCESSED_SUBMAPS = 20;
constexpr size_t CLEANUP_BATCH_SIZE = 10;

/**
 * @brief GNSS data structure for holding georeferenced positions
 */
struct GNSSData {
    double timestamp;                // Timestamp in seconds
    Eigen::Vector3d position;        // Position (x, y, z) in global frame
    Eigen::Matrix3d covariance;      // Full position covariance matrix
    int status;                      // Status/quality indicator (e.g., NavSatFix status)
};

/**
 * @brief Custom factor that constrains the georeference position
 */
class GeoPositionFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
private:
    Eigen::Vector3d measurement_;
    
public:
    GeoPositionFactor(gtsam::Key key1, gtsam::Key key2, 
                     const Eigen::Vector3d& measurement,
                     const gtsam::SharedNoiseModel& model);
                     
    gtsam::Vector evaluateError(const gtsam::Pose3& T_pose, const gtsam::Pose3& X_pose,
                              boost::optional<gtsam::Matrix&> H1 = boost::none,
                              boost::optional<gtsam::Matrix&> H2 = boost::none) const override;
};

/**
 * @brief Module for integrating georeferencing data into the mapping framework
 * 
 * This module accepts various types of georeferencing inputs (PoseStamped, Odometry, NavSatFix)
 * and creates constraints for the global optimization to align the map with a global reference frame.
 */
class Georeference : public ExtensionModuleBase {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Constructor
     */
    Georeference();
    
    /**
     * @brief Destructor
     */
    ~Georeference();

    /**
     * @brief Create ROS topic subscriptions based on configuration
     * @return Vector of topic subscription objects
     */
    virtual std::vector<GenericTopicSubscription::Ptr> create_subscriptions() override;

private:
    /**
     * @brief Handle PoseStamped message callback
     * @param msg Incoming PoseStamped message
     */
    void handle_pose_stamped(const PoseStampedConstPtr& msg);
    
    /**
     * @brief Handle Odometry message callback
     * @param msg Incoming Odometry message
     */
    void handle_odometry(const OdometryConstPtr& msg);
    
    /**
     * @brief Handle NavSatFix message callback
     * @param msg Incoming NavSatFix message
     */
    void handle_nav_sat_fix(const NavSatFixConstPtr& msg);
    
    /**
     * @brief Callback when a new submap is inserted
     * @param submap Pointer to the new submap
     */
    void on_insert_submap(const SubMap::ConstPtr& submap);
    
    /**
     * @brief Callback before global optimization update
     * @param isam2 ISAM2 optimizer
     * @param new_factors New factors to be added
     * @param new_values New values to be added
     */
    void on_smoother_update(
        gtsam_points::ISAM2Ext& isam2,
        gtsam::NonlinearFactorGraph& new_factors,
        gtsam::Values& new_values);
    
    /**
     * @brief Callback after global optimization update
     * @param isam2 ISAM2 optimizer with updated values
     */
    void on_smoother_update_result(gtsam_points::ISAM2Ext& isam2);
    
    /**
     * @brief Background thread for processing data
     */
    void process_data_thread();
    
    /**
     * @brief Process new incoming GNSS and submap data
     * @param local_gnss_data_queue Queue of GNSS data to process
     * @param local_submap_queue Queue of submaps to process
     * @return True if new data was processed
     */
    bool process_new_data(std::deque<GNSSData>& local_gnss_data_queue, 
                          std::deque<SubMap::ConstPtr>& local_submap_queue);
    
    /**
     * @brief Match submaps with their best corresponding GNSS data
     * @param local_gnss_data_queue Queue of available GNSS data
     * @param local_submap_queue Queue of submaps to match
     */
    void match_submaps_with_gnss(std::deque<GNSSData>& local_gnss_data_queue,
                                std::deque<SubMap::ConstPtr>& local_submap_queue);
    
    /**
     * @brief Add georeference factors for matched submap-GNSS pairs
     */
    void add_factors_for_matched_data();
    
    /**
     * @brief Find multiple spatially-distributed GNSS data points for a submap using a 2D grid-based approach
     * @param submap The submap to find GNSS data for
     * @param gnss_queue Queue of available GNSS data points
     * @param grid_cell_size The size of each grid cell for spatial distribution (in meters)
     * @param max_points Maximum number of points to return
     * @return Vector of selected GNSS data points
     */
    std::vector<GNSSData> find_distributed_gnss_points(
        const SubMap::ConstPtr& submap,
        const std::deque<GNSSData>& gnss_queue,
        double grid_cell_size,
        int max_points);
        
    /**
     * @brief Find the best GNSS data point for a given submap
     * @param submap The submap to find data for
     * @param gnss_queue Queue of available GNSS data
     * @param best_gnss Output parameter for the best matching GNSS data
     * @return True if a suitable GNSS data point was found
     */
    bool find_best_gnss_for_submap(const SubMap::ConstPtr& submap, 
                                  const std::deque<GNSSData>& gnss_queue,
                                  GNSSData& best_gnss);
    
    /**
     * @brief Create GNSS data structure from position and covariance
     * @param timestamp Time of the measurement
     * @param position Position vector
     * @param covariance Covariance matrix
     * @param status Status code (default 0)
     * @return Constructed GNSS data object
     */
    GNSSData create_gnss_data(double timestamp, const Eigen::Vector3d& position, 
                             const Eigen::Matrix3d& covariance, int status = 0);
    
    /**
     * @brief Initialize the transformation between local and global coordinates
     * @return True if initialization successful, false otherwise
     */
    bool initialize_transformation();
    
    /**
     * @brief Predict position based on recent measurements
     * @return Predicted position in world frame
     */
    Eigen::Vector3d predict_position();
    
    /**
     * @brief Scale the covariance matrix based on configuration
     * @param covariance Original covariance matrix
     * @return Scaled covariance matrix
     */
    Eigen::Matrix3d scale_covariance(const Eigen::Matrix3d& covariance);
    
    /**
     * @brief Ensure the covariance matrix is valid (positive definite with reasonable values)
     * @param covariance Input covariance matrix
     * @return Regularized covariance matrix
     */
    Eigen::Matrix3d regularize_covariance(const Eigen::Matrix3d& covariance);
    
    /**
     * @brief Scales and regularizes the covariance matrix for numerical stability
     * 
     * First applies the user-configured scaling factor, then ensures the matrix
     * is positive definite with eigenvalues above the minimum threshold.
     * 
     * @param covariance Input covariance matrix
     * @return Prepared covariance matrix
     */
    Eigen::Matrix3d prepare_covariance(const Eigen::Matrix3d& covariance);
    
    /**
     * @brief Add a georeference factor to the optimization
     * @param submap Submap to be constrained
     * @param gnss_data GNSS data to constrain to
     */
    void add_georeference_factor(
        const SubMap::ConstPtr& submap,
        const GNSSData& gnss_data);
        
    /**
     * @brief Sort a container by timestamp
     * @tparam T Container element type (must have timestamp member)
     * @param container Container to sort
     */
    template<typename T>
    void sort_by_timestamp(std::deque<T>& container) {
        std::sort(container.begin(), container.end(),
                 [](const T& a, const T& b) { return a.timestamp < b.timestamp; });
    }

    // Thread management
    std::atomic_bool should_terminate_;
    std::thread processing_thread_;

    // Data queues
    ConcurrentVector<GNSSData> raw_gnss_data_queue_;             // Queue for GNSS data with original covariance
    ConcurrentVector<SubMap::ConstPtr> submap_queue_;
    ConcurrentVector<gtsam::NonlinearFactor::shared_ptr> factor_queue_;

    // Processed data storage
    std::vector<SubMap::ConstPtr> processed_submaps_;
    std::deque<GNSSData> submap_best_gnss_data_;       // Stores the best GNSSData point for each processed submap
    std::unordered_set<int> processed_submap_ids_;  // Track which submaps we've already processed

    // Configuration parameters
    std::string input_topic_;
    std::string input_type_;
    Eigen::Vector3d prior_inf_scale_;
    double min_baseline_;
    int min_init_factors_;                         // Minimum number of factors needed before initialization
    
    // Covariance handling parameters
    double covariance_scale_;
    double min_covariance_eigenvalue_;
    double position_prediction_window_;
    
    // Grid-based point selection parameters
    double grid_cell_size_;
    int max_gnss_points_per_submap_;
    
    // Frame transformation parameters
    std::string lidar_frame_id_;
    std::string gps_frame_id_;
    Eigen::Isometry3d T_lidar_gps_;  // Transform from GPS frame to LiDAR frame
    bool manual_transform_provided_ = false;

    // State tracking
    bool transformation_initialized_;
    bool first_factors_added_ = false;
    Eigen::Isometry3d T_world_local_coord_;  // Transform from local coordinates to world frame
    Eigen::Isometry3d T_geo_world_;          // Transform from world frame to geo frame

    // Logging
    std::shared_ptr<spdlog::logger> logger_;
};

}  // namespace glim

// Module factory function
extern "C" glim::ExtensionModule* create_extension_module() {
    return new glim::Georeference();
}
