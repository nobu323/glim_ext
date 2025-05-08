#include <deque>
#include <atomic>
#include <thread>
#include <numeric>
#include <Eigen/Core>

#define GLIM_ROS2

#include <boost/format.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/concurrent_vector.hpp>

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

template <typename Stamp>
double to_sec(const Stamp& stamp) {
  return stamp.sec + stamp.nanosec / 1e9;
}
#else
#include <glim/util/extension_module_ros.hpp>
#include <geometry_msgs/PoseWithCovarianceStamped.hpp>

using ExtensionModuleBase = glim::ExtensionModuleROS;
#endif

#include <spdlog/spdlog.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/Expression.h>

#include <glim/util/logging.hpp>
#include <glim/util/convert_to_string.hpp>
#include <glim_ext/util/config_ext.hpp>
#include <glim_ext/geodetic.hpp>

namespace glim {

struct GNSSData {
  double stamp;
  Eigen::Vector3d pos;
  Eigen::Vector3d cov;
};

using gtsam::symbol_shorthand::X;

/**
 * @brief Module for incorporating georeferencing data (PoseStamped, Odometry, NavSatFix) into global optimization.
 */
class Georeference : public ExtensionModuleBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Georeference() : logger(create_module_logger("georeference")) {
    logger_->info("initializing georeference module");
    const std::string config_path = glim::GlobalConfigExt::get_config_path("config_georeference");
    logger_->info("georeference_config_path={}", config_path);

    glim::Config config(config_path);
    input_topic_ = config.param<std::string>("georeference", "input_topic", "/georeference_input");
    input_type_ = config.param<std::string>("georeference", "input_type", "PoseStamped");  // PoseStamped, Odometry, NavSatFix
    prior_inf_scale_ = config.param<Eigen::Vector3d>("georeference", "prior_inf_scale", Eigen::Vector3d(1e3, 1e3, 0.0));
    min_baseline_ = config.param<double>("georeference", "min_baseline", 5.0);

    transformation_initialized_ = false;
    T_world_local_coord_.setIdentity();

    kill_switch_ = false;
    thread_ = std::thread([this] { backend_task(); });

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    GlobalMappingCallbacks::on_insert_submap.add(std::bind(&Georeference::on_insert_submap, this, _1));
    GlobalMappingCallbacks::on_smoother_update.add(std::bind(&Georeference::on_smoother_update, this, _1, _2, _3));
    GlobalMappingCallbacks::on_smoother_update_result.add(std::bind(&Georeference::on_smoother_update_result, this, _1));
  }
  ~Georeference() {
    kill_switch_ = true;
    thread_.join();
  }

  virtual std::vector<GenericTopicSubscription::Ptr> create_subscriptions() override {
    std::vector<GenericTopicSubscription::Ptr> subs;
    if (input_type_ == "PoseStamped") {
      subs.push_back(std::make_shared<TopicSubscription<PoseStamped>>(input_topic_, [this](const PoseStampedConstPtr msg) { georeference_callback(msg); }));
    } else if (input_type_ == "Odometry") {
      subs.push_back(std::make_shared<TopicSubscription<Odometry>>(input_topic_, [this](const OdometryConstPtr msg) { georeference_callback(msg); }));
    } else if (input_type_ == "NavSatFix") {
      subs.push_back(std::make_shared<TopicSubscription<NavSatFix>>(input_topic_, [this](const NavSatFixConstPtr msg) { georeference_callback(msg); }));
    } else {
      logger_->error("Unknown georeference input_type_: {}", input_type_);
    }
    return subs;
  }

  void georeference_callback(const PoseStampedConstPtr& msg) {
    Eigen::Vector4d data;
    const double stamp = to_sec(msg->header.stamp);
    const auto& pos = msg->pose.position;
    data << stamp, pos.x, pos.y, pos.z;
    input_coord_queue_.push_back(data);
  }

  void georeference_callback(const OdometryConstPtr& msg) {
    Eigen::Vector4d data;
    const double stamp = to_sec(msg->header.stamp);
    const auto& pos = msg->pose.pose.position;
    data << stamp, pos.x, pos.y, pos.z;
    input_coord_queue_.push_back(data);
  }

  void georeference_callback(const NavSatFixConstPtr& msg) {
    Eigen::Vector4d data;
    const double stamp = to_sec(msg->header.stamp);
    const Eigen::Vector3d ecef = wgs84_to_ecef(msg->latitude, msg->longitude, msg->altitude);
    data << stamp, ecef.x(), ecef.y(), ecef.z();
    input_coord_queue_.push_back(data);
  }

  void on_insert_submap(const SubMap::ConstPtr& submap) { input_submap_queue_.push_back(submap); }

  void on_smoother_update(gtsam_points::ISAM2Ext& isam2, gtsam::NonlinearFactorGraph& new_factors, gtsam::Values& new_values) {
    if (!isam2.valueExists(T(0))) {
      // insert T_geo_world variable
      new_values.insert(T(0), gtsam::Pose3::Identity());
    }

    static bool first_factors = true;
    if (first_factors) {
      if (output_factors__.size() < 10) {
        // wait until we have enough factors to prevent underconstrained system
        return;
      }
      first_factors = false;
    }

    const auto factors = output_factors__.get_all_and_clear();
    if (!factors.empty()) {
      logger_->debug("insert {} georeferencing prior factors", factors.size());
      new_factors.add(factors);
    }
  }

  void on_smoother_update_result(gtsam_points::ISAM2Ext& isam2) {
    T_geo_world_ = Eigen::Isometry3d(isam2.calculateEstimate<gtsam::Pose3>(T(0)).matrix());
    logger_->info("estimated T_geo_world={}", convert_to_string(T_geo_world_));
  }

  void backend_task() {
    logger_->info("starting georeference backend thread");
    std::deque<Eigen::Vector4d> coord_queue;  // Stores [stamp, x, y, z]
    std::deque<SubMap::ConstPtr> submap_queue;

    while (!kill_switch_) {
      // Get new coordinate data
      const auto coord_data = input_coord_queue_.get_all_and_clear();
      coord_queue.insert(coord_queue.end(), coord_data.begin(), coord_data.end());

      // Add new submaps
      const auto new_submaps = input_submap_queue_.get_all_and_clear();
      if (new_submaps.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
      }
      submap_queue.insert(submap_queue.end(), new_submaps.begin(), new_submaps.end());

      // Remove submaps that are created earlier than the oldest coordinate data
      while (!coord_queue.empty() && !submap_queue.empty() && submap_queue.front()->frames.front()->stamp < coord_queue.front()[0]) {
        submap_queue.pop_front();
      }

      // Interpolate coordinate data and associate with submaps
      while (!coord_queue.empty() && !submap_queue.empty() && submap_queue.front()->frames.front()->stamp > coord_queue.front()[0] &&
             submap_queue.front()->frames.back()->stamp < coord_queue.back()[0]) {
        const auto& submap = submap_queue.front();
        const double stamp = submap->frames[submap->frames.size() / 2]->stamp;

        const auto right = std::lower_bound(coord_queue.begin(), coord_queue.end(), stamp, [](const Eigen::Vector4d& coord, const double t) { return coord[0] < t; });
        if (right == coord_queue.end() || (right + 1) == coord_queue.end()) {
          logger_->warn("invalid condition in georeference module!!");
          break;
        }
        const auto left = right - 1;
        logger_->debug("submap={:.6f} coord_left={:.6f} coord_right={:.6f}", stamp, (*left)[0], (*right)[0]);

        const double tl = (*left)[0];
        const double tr = (*right)[0];
        const double p = (stamp - tl) / (tr - tl);
        const Eigen::Vector4d interpolated = (1.0 - p) * (*left) + p * (*right);

        submaps_.push_back(submap);
        submap_coords_.push_back(interpolated);

        submap_queue.pop_front();
        coord_queue.erase(coord_queue.begin(), left);
      }

      // Initialize T_world_local_coord
      if (!transformation_initialized_ && !submaps.empty() && (submaps.front()->T_world_origin.inverse() * submaps_.back()->T_world_origin).translation().norm() > min_baseline_) {
        Eigen::Vector3d mean_est = Eigen::Vector3d::Zero();
        Eigen::Vector3d mean_coord = Eigen::Vector3d::Zero();
        for (int i = 0; i < submaps_.size(); i++) {
          mean_est += submaps_[i]->T_world_origin.translation();
          mean_coord += submap_coords_[i].tail<3>();
        }
        mean_est /= submaps_.size();
        mean_coord /= submaps_.size();

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (int i = 0; i < submaps_.size(); i++) {
          const Eigen::Vector3d centered_est = submaps_[i]->T_world_origin.translation() - mean_est;
          const Eigen::Vector3d centered_coord = submap_coords_[i].tail<3>() - mean_coord;
          cov += centered_coord * centered_est.transpose();
        }
        cov /= submaps_.size();

        const Eigen::JacobiSVD<Eigen::Matrix2d> svd(cov.block<2, 2>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Matrix2d U = svd.matrixU();
        const Eigen::Matrix2d V = svd.matrixV();
        const Eigen::Matrix2d D = svd.singularValues().asDiagonal();
        Eigen::Matrix2d S = Eigen::Matrix2d::Identity();

        const double det = U.determinant() * V.determinant();
        if (det < 0.0) {
          S(1, 1) = -1;
        }

        Eigen::Isometry3d T_local_coord_world = Eigen::Isometry3d::Identity();
        T_local_coord_world.linear().block<2, 2>(0, 0) = U * S * V.transpose();
        T_local_coord_world.translation() = mean_coord - T_local_coord_world.linear() * mean_est;

        T_world_local_coord_ = T_local_coord_world.inverse();

        for (int i = 0; i < submaps_.size(); i++) {
          const Eigen::Vector3d coord_in_world = T_world_local_coord_ * submap_coords_[i].tail<3>();
          logger_->debug("submap={} coord_in_world={}", convert_to_string(submaps[i]->T_world_origin.translation().eval()), convert_to_string(coord_in_world));
        }

        logger_->info("T_world_local_coord={}", convert_to_string(T_world_local_coord_));
        transformation_initialized_ = true;
      }

      // Add translation prior factor
      if (!submap_coords.empty() && !submaps.empty()) {
        const auto& submap = submaps.back();
        const auto& coords = submap_coords.back();

        // Create a GNSSData structure to hold the interpolated data
        GNSSData interpolated = {
          .stamp = coords[0],
          .pos = coords.tail<3>(),
          // You'll need to create a covariance matrix - using prior_inf_scale for now
          .cov = (Eigen::Vector3d(1.0, 1.0, 1.0).array() / prior_inf_scale.array()).matrix()};

        // Create expressions for the transformation
        gtsam::Expression<gtsam::Pose3> T_geo_world(T(0));
        gtsam::Expression<gtsam::Pose3> world_pose(X(submap->id));
        auto geo_p = gtsam::translation(T_geo_world * world_pose);

        // Add GNSS measurement factor
        logger->info(
          "Add Georeferencing factor between submap={} and gnss={}",
          convert_to_string(submap->T_world_origin.translation().eval()),
          convert_to_string(interpolated.pos));

        auto noise_model = gtsam::noiseModel::Diagonal::Variances(interpolated.cov);
        gtsam::NonlinearFactor::shared_ptr factor(new gtsam::ExpressionFactor<gtsam::Point3>(noise_model, interpolated.pos, geo_p));
        output_factors_.push_back(factor);
      }
    }
  }

private:
  std::atomic_bool kill_switch_;
  std::thread thread_;

  ConcurrentVector<Eigen::Vector4d> input_coord_queue_;
  ConcurrentVector<SubMap::ConstPtr> input_submap_queue_;
  ConcurrentVector<gtsam::NonlinearFactor::shared_ptr> output_factors_;

  std::vector<SubMap::ConstPtr> submaps_;
  std::vector<Eigen::Vector4d> submap_coords_;

  std::string input_topic_;
  std::string input_type_;
  Eigen::Vector3d prior_inf_scale_;
  double min_baseline_;

  bool transformation_initialized_;
  Eigen::Isometry3d T_world_local_coord_;
  Eigen::Isometry3d T_geo_world_;

  // Logging
  std::shared_ptr<spdlog::logger> logger_;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::GNSSGlobal();
}
