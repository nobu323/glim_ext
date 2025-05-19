#include <glim_ext/georeference_module.hpp>
#include <gtsam_points/optimizers/isam2_ext.hpp> 
#include <unordered_set>

namespace glim {

/**
 * Calculate the average separation distance between GNSS points
 * @param points Vector of GNSS data points
 * @return Average separation distance in meters
 */
double Georeference::calculate_average_separation(const std::vector<GNSSData>& points) {
    if (points.size() < 2) {
        return 0.0;
    }
    
    double total_distance = 0.0;
    int count = 0;
    
    // Calculate distances between all unique pairs
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            total_distance += (points[i].position - points[j].position).norm();
            count++;
        }
    }
    
    return count > 0 ? total_distance / count : 0.0;
}

/**
 * Helper function for calculating skew symmetric matrix
 * @param wx X component
 * @param wy Y component
 * @param wz Z component
 * @return Skew symmetric matrix
 */
inline gtsam::Matrix33 skewSymmetric(double wx, double wy, double wz) {
    gtsam::Matrix33 skew;
    skew << 0, -wz, wy,
            wz, 0, -wx,
            -wy, wx, 0;
    return skew;
}

/**
 * Implementation of GeoPositionFactor constructor
 * @param key1 Key for the global transform pose ('T')
 * @param key2 Key for the submap pose ('X')
 * @param measurement Measurement vector (global position)
 * @param model Noise model
 */
GeoPositionFactor::GeoPositionFactor(gtsam::Key key1, gtsam::Key key2, 
                                   const Eigen::Vector3d& measurement,
                                   const gtsam::SharedNoiseModel& model)
    : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, key1, key2),
      measurement_(measurement) {}
      
gtsam::Vector GeoPositionFactor::evaluateError(
    const gtsam::Pose3& T_pose, const gtsam::Pose3& X_pose,
    boost::optional<gtsam::Matrix&> H1, 
    boost::optional<gtsam::Matrix&> H2) const 
{
    // Compose the two poses: T * X
    gtsam::Pose3 composed = T_pose.compose(X_pose);
    
    // Extract the position component
    gtsam::Point3 position = composed.translation();
    
    // Calculate error
    gtsam::Vector3 error = position - measurement_;
    
    // Calculate Jacobians if requested
    if (H1 || H2) {
        // For avoiding underconstrained systems, we need precise Jacobians that
        // only affect the position components of the poses
        
        // Position Jacobian wrt T
        if (H1) {
            *H1 = gtsam::Matrix::Zero(3, 6);
            
            // For rotation part: We use zeros to avoid constraining rotation
            // This effectively makes the factor only constrain the translation
            H1->block<3,3>(0,0) = gtsam::Matrix3::Zero();
            
            // For translation part: Identity matrix as translation directly affects position
            H1->block<3,3>(0,3) = gtsam::Matrix3::Identity();
        }
        
        // Position Jacobian wrt X
        if (H2) {
            *H2 = gtsam::Matrix::Zero(3, 6);
            
            // For rotation part: We use zeros to avoid constraining rotation
            // This effectively makes the factor only constrain the translation
            H2->block<3,3>(0,0) = gtsam::Matrix3::Zero();
            
            // For translation part: Only affected by T's rotation matrix
            H2->block<3,3>(0,3) = T_pose.rotation().matrix();
        }
    }
    
    return error;
}

Georeference::Georeference() 
    : logger_(create_module_logger("georeference")),
      should_terminate_(false),
      transformation_initialized_(false),
      first_factors_added_(false)
{
    logger_->info("Initializing georeference module");
    
    // Load configuration
    const std::string config_path = glim::GlobalConfigExt::get_config_path("config_georeference");
    logger_->info("Georeference config path: {}", config_path);

    glim::Config config(config_path);
    // Basic configuration
    input_topic_ = config.param<std::string>("georeference", "input_topic", "/georeference_input");
    input_type_ = config.param<std::string>("georeference", "input_type", "PoseStamped");
    prior_inf_scale_ = config.param<Eigen::Vector3d>("georeference", "prior_inf_scale", Eigen::Vector3d(1e3, 1e3, 1e3));
    min_baseline_ = config.param<double>("georeference", "min_baseline", 5.0);
    min_init_factors_ = config.param<int>("georeference", "min_init_factors", 3);
    
    // Multi-GNSS point selection parameters
    max_gnss_points_per_submap_ = config.param<int>("georeference", "max_gnss_points_per_submap", 5);
    min_gnss_point_separation_ = config.param<double>("georeference", "min_gnss_point_separation", 2.0);
    
    logger_->info("Georeference multi-point configuration: max_points_per_submap={}, min_separation={:.2f}m",
                 max_gnss_points_per_submap_, min_gnss_point_separation_);
    
    // Covariance handling parameters
    covariance_scale_ = config.param<double>("georeference", "covariance_scale", 10.0);
    min_covariance_eigenvalue_ = config.param<double>("georeference", "min_covariance_eigenvalue", 1e-3);
    position_prediction_window_ = config.param<double>("georeference", "position_prediction_window", 5.0);
    
    // Frame transformation parameters
    lidar_frame_id_ = config.param<std::string>("georeference", "lidar_frame_id", "lidar");
    gps_frame_id_ = config.param<std::string>("georeference", "gps_frame_id", "gps");
    
    // Default identity transform
    T_lidar_gps_.setIdentity();
    
    // Load frame transform from config
    Eigen::Vector3d translation = config.param<Eigen::Vector3d>(
        "georeference", "lidar_gps_translation", Eigen::Vector3d::Zero());
        
    Eigen::Vector3d rpy = config.param<Eigen::Vector3d>(
        "georeference", "lidar_gps_rotation_rpy", Eigen::Vector3d::Zero());
    
    // Check if a non-default transform was specified
    if (!translation.isZero() || !rpy.isZero()) {
        T_lidar_gps_.translation() = translation;
        T_lidar_gps_.linear() = Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
                              Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
                              Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()).toRotationMatrix();
                              
        manual_transform_provided_ = true;
        logger_->info("Using manually configured LiDAR-GPS transform: {}", convert_to_string(T_lidar_gps_));
    }

    // Initialize transformations
    T_world_local_coord_.setIdentity();
    T_geo_world_.setIdentity();

    // Register callbacks
    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    GlobalMappingCallbacks::on_insert_submap.add(std::bind(&Georeference::on_insert_submap, this, _1));
    GlobalMappingCallbacks::on_smoother_update.add(std::bind(&Georeference::on_smoother_update, this, _1, _2, _3));
    GlobalMappingCallbacks::on_smoother_update_result.add(std::bind(&Georeference::on_smoother_update_result, this, _1));
    
    // Start processing thread
    processing_thread_ = std::thread(&Georeference::process_data_thread, this);
    logger_->info("Georeference module initialized");
}

Georeference::~Georeference() {
    logger_->info("Shutting down georeference module");
    should_terminate_ = true;
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

std::vector<GenericTopicSubscription::Ptr> Georeference::create_subscriptions() {
    std::vector<GenericTopicSubscription::Ptr> subscriptions;
    
    logger_->info("Creating subscription to {} with type {}", input_topic_, input_type_);
    
    if (input_type_ == "PoseStamped") {
        subscriptions.push_back(std::make_shared<TopicSubscription<PoseStamped>>(
            input_topic_, 
            [this](const PoseStampedConstPtr& msg) { handle_pose_stamped(msg); }
        ));
    } else if (input_type_ == "Odometry") {
        subscriptions.push_back(std::make_shared<TopicSubscription<Odometry>>(
            input_topic_, 
            [this](const OdometryConstPtr& msg) { handle_odometry(msg); }
        ));
    } else if (input_type_ == "NavSatFix") {
        subscriptions.push_back(std::make_shared<TopicSubscription<NavSatFix>>(
            input_topic_, 
            [this](const NavSatFixConstPtr& msg) { handle_nav_sat_fix(msg); }
        ));
    } else {
        logger_->error("Unknown input type: {}. Valid types are PoseStamped, Odometry, or NavSatFix", input_type_);
    }
    
    return subscriptions;
}

GNSSData Georeference::create_gnss_data(double timestamp, const Eigen::Vector3d& position, 
                                       const Eigen::Matrix3d& covariance, int status) {
    GNSSData data;
    data.timestamp = timestamp;
    data.position = position;
    data.covariance = covariance;
    data.status = status;
    return data;
}

void Georeference::handle_pose_stamped(const PoseStampedConstPtr& msg) {
    const double timestamp = to_sec(msg->header.stamp);
    const auto& position = msg->pose.position;
    
    // Create a default diagonal covariance matrix for PoseStamped
    // since PoseStamped doesn't include covariance information
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
    
    // Create and store GNSS data object
    GNSSData gnss_data = create_gnss_data(
        timestamp, 
        Eigen::Vector3d(position.x, position.y, position.z),
        covariance
    );
    
    raw_gnss_data_queue_.push_back(std::move(gnss_data));
    
    logger_->debug("Received PoseStamped at t={:.3f}", timestamp);
}

void Georeference::handle_odometry(const OdometryConstPtr& msg) {
    const double timestamp = to_sec(msg->header.stamp);
    const auto& position = msg->pose.pose.position;
    
    // Extract covariance matrix for position
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
    if (msg->pose.covariance.size() >= 36) { // 6x6 covariance matrix
        // Extract the position part (top-left 3x3 submatrix)
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                covariance(i, j) = msg->pose.covariance[i * 6 + j];
            }
        }
    }
    
    // Create and store GNSS data object
    GNSSData gnss_data = create_gnss_data(
        timestamp, 
        Eigen::Vector3d(position.x, position.y, position.z),
        covariance
    );
    
    raw_gnss_data_queue_.push_back(std::move(gnss_data));
    
    logger_->debug("Received Odometry at t={:.3f}", timestamp);
}

void Georeference::handle_nav_sat_fix(const NavSatFixConstPtr& msg) {
    // Filter out messages with low-quality fixes (status < 2)
    if (msg->status.status < 2) {
        logger_->debug("Ignoring NavSatFix message with insufficient fix quality: status={}", msg->status.status);
        return;
    }

    const double timestamp = to_sec(msg->header.stamp);
    const Eigen::Vector3d ecef = wgs84_to_ecef(msg->latitude, msg->longitude, msg->altitude);
    
    // Extract covariance matrix
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
    if (msg->position_covariance_type != 0) { // If covariance is provided
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                covariance(i, j) = msg->position_covariance[i * 3 + j];
            }
        }
        
        // Convert ENU covariance to ECEF covariance
        // This is a simplified conversion - a proper conversion would involve the Jacobian
        // of the ENU to ECEF transformation at the current location
        // TODO: Implement proper ENU to ECEF covariance transformation
    }
    
    // Create and store GNSS data object
    GNSSData gnss_data = create_gnss_data(
        timestamp, 
        ecef,
        covariance,
        msg->status.status
    );
    
    raw_gnss_data_queue_.push_back(std::move(gnss_data));
    
    logger_->debug("Received NavSatFix at t={:.3f} lat={:.6f} lon={:.6f} alt={:.2f} status={} cov_type={}", 
                  timestamp, msg->latitude, msg->longitude, msg->altitude, 
                  msg->status.status, msg->position_covariance_type);
}

void Georeference::on_insert_submap(const SubMap::ConstPtr& submap) {
    submap_queue_.push_back(submap);
    logger_->debug("Submap {} inserted at t={:.3f}", submap->id, submap->frames.front()->stamp);
}

/**
 * Callback before global optimization update
 * @param isam2 ISAM2 optimizer
 * @param new_factors New factors to be added to the optimization
 * @param new_values New values to be added to the optimization
 */
void Georeference::on_smoother_update(
    gtsam_points::ISAM2Ext& isam2, 
    gtsam::NonlinearFactorGraph& new_factors, 
    gtsam::Values& new_values) 
{
    // Create symbol/key for the T transformation
    gtsam::Key T_key = gtsam::Symbol('T', 0);
    
    // Initialize geo transformation variable if not exists
    if (!isam2.valueExists(T_key)) {
        logger_->info("Initializing T_geo_world variable in optimization");
        new_values.insert(T_key, gtsam::Pose3::Identity());
    }

    // Using 'X' symbol for submaps as defined in the header file

    // Add accumulated factors to optimization
    const auto factors = factor_queue_.get_all_and_clear();
    if (!factors.empty()) {
        logger_->info("Adding {} georeference factors to optimization", factors.size());
        
        // Create a filtered list of factors that only reference existing keys
        gtsam::NonlinearFactorGraph filtered_factors;
        for (const auto& factor : factors) {
            // Check if all keys referenced by this factor exist in the optimizer
            bool all_keys_exist = true;
            for (const auto& key : factor->keys()) {
                if (!isam2.valueExists(key)) {
                    all_keys_exist = false;
                    
                    // Log the non-existent key
                    char symbol = gtsam::symbolChr(key);
                    auto index = gtsam::symbolIndex(key);
                    logger_->warn("Skipping factor with non-existent key: '{}' {}", symbol, index);
                    break;
                }
            }
            
            if (all_keys_exist) {
                filtered_factors.add(factor);
            }
        }
        
        // Add only the filtered factors
        if (filtered_factors.size() > 0) {
            logger_->info("Adding {} valid georeference factors (skipped {})", 
                        filtered_factors.size(), factors.size() - filtered_factors.size());
            new_factors.add(filtered_factors);
        } else {
            logger_->warn("No valid georeference factors to add (all keys missing)");
        }
    }
}

/**
 * Callback after global optimization update
 * @param isam2 ISAM2 optimizer with updated values
 */
void Georeference::on_smoother_update_result(gtsam_points::ISAM2Ext& isam2) {
    // Create symbol/key for the T transformation
    gtsam::Key T_key = gtsam::Symbol('T', 0);
    
    if (isam2.valueExists(T_key)) {
        // Get the new estimate from optimizer
        Eigen::Isometry3d new_T_geo_world = Eigen::Isometry3d(isam2.calculateEstimate<gtsam::Pose3>(T_key).matrix());
        
        // Check if transform has changed from previous value
        bool has_changed = false;
        
        // Compare with epsilon for floating point values
        const double epsilon = 1e-6;
        const Eigen::Vector3d old_trans = T_geo_world_.translation();
        const Eigen::Vector3d new_trans = new_T_geo_world.translation();
        const Eigen::Quaterniond old_rot(T_geo_world_.rotation());
        const Eigen::Quaterniond new_rot(new_T_geo_world.rotation());
        
        // Check if translation has changed significantly
        if ((old_trans - new_trans).norm() > epsilon) {
            has_changed = true;
        }
        
        // Check if rotation has changed significantly (via quaternion dot product)
        // Two quaternions are identical when their dot product is +/-1
        if (std::abs(std::abs(old_rot.dot(new_rot)) - 1.0) > epsilon) {
            has_changed = true;
        }
        
        // Update the stored transform
        T_geo_world_ = new_T_geo_world;
        
        // Only log if there was a meaningful change
        if (has_changed) {
            logger_->info("T_geo_world updated: {}", convert_to_string(T_geo_world_));
        } else if (new_trans.norm() < epsilon && std::abs(std::abs(new_rot.w()) - 1.0) < epsilon) {
            // Log a special case for identity transform to help with debugging
            static bool logged_zero_warning = false;
            if (!logged_zero_warning) {
                logger_->warn("T_geo_world remains at identity transform - this may indicate initialization issues");
                logged_zero_warning = true;
            }
        }
    }
}

Eigen::Matrix3d Georeference::scale_covariance(const Eigen::Matrix3d& covariance) {
    // Scale the covariance by the user-configurable factor
    return covariance * covariance_scale_;
}

Eigen::Matrix3d Georeference::regularize_covariance(const Eigen::Matrix3d& covariance) {
    // Ensure the covariance matrix is valid (positive definite with reasonable values)
    
    // Perform eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covariance);
    Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
    Eigen::Matrix3d eigenvectors = eigensolver.eigenvectors();
    
    // Ensure minimum eigenvalues for numerical stability
    for (int i = 0; i < 3; ++i) {
        eigenvalues(i) = std::max(eigenvalues(i), min_covariance_eigenvalue_);
    }
    
    // Reconstruct covariance matrix with regularized eigenvalues
    return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}

Eigen::Matrix3d Georeference::prepare_covariance(const Eigen::Matrix3d& covariance) {
    // First scale the covariance
    Eigen::Matrix3d scaled = scale_covariance(covariance);
    
    // Then regularize it
    Eigen::Matrix3d regularized = regularize_covariance(scaled);
    
    // Apply a small additional multiplier to position constraints
    // to ensure they don't overly constrain the system
    // This helps prevent numerical issues in the optimizer
    const double position_constraint_weight = 10.0; // Adjust if needed
    return regularized * position_constraint_weight;
}

Eigen::Vector3d Georeference::predict_position() {
    // For ECEF coordinates, simple prediction can be challenging due to the large scale
    // and the fact that the robot's motion is typically on a small local scale
    
    // If we don't have transformation yet, just use the latest submap position
    if (!transformation_initialized_) {
        if (processed_submaps_.empty()) {
            return Eigen::Vector3d::Zero();
        }
        return processed_submaps_.back()->T_world_origin.translation().eval();
    }
    
    // If transformation is initialized, try to use the latest GNSS measurement
    // Since we're in the global ECEF frame now
    if (!processed_submaps_.empty()) {
        const auto& last_submap = processed_submaps_.back();
        const auto& gnss_points = submap_gnss_data_map_[last_submap->id];
        
        if (!gnss_points.empty()) {
            // Use the first (best quality) GNSS point position
            return gnss_points[0].position;
        }
    }
    
    // Fallback to zero if no data available
    return Eigen::Vector3d::Zero();
}

bool Georeference::initialize_transformation() {
    // Check if we have enough data for transformation initialization
    if (processed_submaps_.size() < 2) {
        logger_->debug("Not enough processed submaps ({}) for initialization, need at least 2", 
                     processed_submaps_.size());
        return false;
    }
    
    // Check if we have sufficient baseline distance
    const auto& first_submap = processed_submaps_.front();
    const auto& last_submap = processed_submaps_.back();
    
    const Eigen::Isometry3d relative_motion = 
        first_submap->T_world_origin.inverse() * last_submap->T_world_origin;
    
    const double baseline = relative_motion.translation().norm();
    if (baseline <= min_baseline_) {
        logger_->debug("Baseline distance ({:.2f}m) too small for initialization, needs > {:.2f}m", 
                      baseline, min_baseline_);
        return false;
    }
    
    // Calculate centroids
    Eigen::Vector3d centroid_est = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid_coord = Eigen::Vector3d::Zero();
    int total_points = 0;
    
    for (const auto& submap : processed_submaps_) {
        const auto& gnss_points = submap_gnss_data_map_[submap->id];
        if (gnss_points.empty()) continue;
        
        centroid_est += submap->T_world_origin.translation() * gnss_points.size();
        
        for (const auto& gnss : gnss_points) {
            centroid_coord += gnss.position;
        }
        
        total_points += gnss_points.size();
    }
    
    if (total_points == 0) {
        logger_->error("No GNSS points available for initialization");
        return false;
    }
    
    centroid_est /= total_points;
    centroid_coord /= total_points;

    // Calculate covariance matrix
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto& submap : processed_submaps_) {
        const auto& gnss_points = submap_gnss_data_map_[submap->id];
        if (gnss_points.empty()) continue;
        
        const Eigen::Vector3d submap_pos = submap->T_world_origin.translation();
        const Eigen::Vector3d centered_est = submap_pos - centroid_est;
        
        for (const auto& gnss : gnss_points) {
            const Eigen::Vector3d centered_coord = gnss.position - centroid_coord;
            covariance += centered_coord * centered_est.transpose();
        }
    }
    covariance /= total_points;

    // SVD for 2D alignment (using only x,y components)
    const Eigen::JacobiSVD<Eigen::Matrix2d> svd(
        covariance.block<2, 2>(0, 0), 
        Eigen::ComputeFullU | Eigen::ComputeFullV
    );
    
    const Eigen::Matrix2d U = svd.matrixU();
    const Eigen::Matrix2d V = svd.matrixV();
    Eigen::Matrix2d S = Eigen::Matrix2d::Identity();

    // Correct for possible reflection
    const double det = U.determinant() * V.determinant();
    if (det < 0.0) {
        S(1, 1) = -1.0;
    }

    // Construct transformation
    Eigen::Isometry3d T_local_coord_world = Eigen::Isometry3d::Identity();
    T_local_coord_world.linear().block<2, 2>(0, 0) = U * S * V.transpose();
    T_local_coord_world.translation() = centroid_coord - T_local_coord_world.linear() * centroid_est;

    T_world_local_coord_ = T_local_coord_world.inverse();
    logger_->info("Transformation initialized: T_world_local_coord = {}", convert_to_string(T_world_local_coord_));

    // Log validation info
    for (const auto& submap : processed_submaps_) {
        const auto& gnss_points = submap_gnss_data_map_[submap->id];
        if (gnss_points.empty()) continue;
        
        const Eigen::Vector3d submap_pos = submap->T_world_origin.translation();
        
        // Log average error for this submap
        double total_error = 0.0;
        for (const auto& gnss : gnss_points) {
            const Eigen::Vector3d coord_in_world = T_world_local_coord_ * gnss.position;
            const double error = (coord_in_world - submap_pos).norm();
            total_error += error;
        }
        
        double avg_error = total_error / gnss_points.size();
        logger_->debug("Submap {}: estimated={}, average error={:.2f}m from {} GNSS points", 
                     submap->id,
                     convert_to_string(submap_pos.eval()),
                     avg_error,
                     gnss_points.size());
    }

    return true;
}

/**
 * Find the best GNSS data point for a given submap based on timestamps and covariance quality
 * @param submap The submap to find GNSS data for
 * @param gnss_queue Queue of available GNSS data points
 * @param best_gnss Output parameter for the best matching GNSS data
 * @return True if a suitable GNSS data point was found, false otherwise
 */
/**
 * Find multiple spatially-distributed GNSS data points for a submap
 * @param submap The submap to find GNSS data for
 * @param gnss_queue Queue of available GNSS data points
 * @return Vector of selected GNSS data points
 */
std::vector<GNSSData> Georeference::find_distributed_gnss_points(
    const SubMap::ConstPtr& submap,
    const std::deque<GNSSData>& gnss_queue)
{
    // Get submap time window
    const auto& frames = submap->frames;
    const double submap_start_time = frames.front()->stamp;
    const double submap_end_time = frames.back()->stamp;
    
    // Find candidate GNSS points within the submap's time window
    std::vector<GNSSData> candidates;
    for (const auto& gnss_data : gnss_queue) {
        if (gnss_data.timestamp >= submap_start_time && 
            gnss_data.timestamp <= submap_end_time) {
            // Add only candidates with valid covariance
            if (gnss_data.covariance.allFinite() && 
                gnss_data.covariance.trace() > MINIMUM_COVARIANCE_TRACE) {
                candidates.push_back(gnss_data);
            }
        }
        // Early exit optimization
        if (gnss_data.timestamp > submap_end_time && !candidates.empty()) {
            break;
        }
    }

    if (candidates.empty()) {
        logger_->warn("No valid GNSS candidates found for submap {} (t=[{:.3f}-{:.3f}])",
                   submap->id, submap_start_time, submap_end_time);
        return {};
    }

    // Sort candidates by quality (lower trace = better precision)
    std::sort(candidates.begin(), candidates.end(), 
        [](const GNSSData& a, const GNSSData& b) {
            return a.covariance.trace() < b.covariance.trace();
        });
    
    // Select spatially distributed points
    std::vector<GNSSData> selected;
    selected.push_back(candidates[0]); // Always take the best one
    
    // Try to add more points that are well-separated
    for (size_t i = 1; i < candidates.size() && 
         selected.size() < static_cast<size_t>(max_gnss_points_per_submap_); i++) {
        
        bool is_well_separated = true;
        // Check distance to all previously selected points
        for (const auto& sel_point : selected) {
            double distance = (candidates[i].position - sel_point.position).norm();
            if (distance < min_gnss_point_separation_) {
                is_well_separated = false;
                break;
            }
        }
        
        if (is_well_separated) {
            selected.push_back(candidates[i]);
        }
    }
    
    logger_->info("Selected {} GNSS points for submap {} from {} candidates",
                selected.size(), submap->id, candidates.size());
                 
    return selected;
}

void Georeference::add_georeference_factor(
    const SubMap::ConstPtr& submap,
    const GNSSData& gnss_data)
{
    // Apply LiDAR-GPS transform to the GNSS data if manually configured
    Eigen::Vector3d transformed_position = gnss_data.position;
    Eigen::Matrix3d transformed_covariance = gnss_data.covariance;
    
    // Only apply transform if manual transform is provided
    if (manual_transform_provided_) {
        // Transform position
        transformed_position = T_lidar_gps_ * transformed_position;
        
        // Transform covariance
        transformed_covariance = T_lidar_gps_.rotation() * transformed_covariance * T_lidar_gps_.rotation().transpose();
        
        logger_->debug("Applied LiDAR-GPS transform to GNSS data. Original: {}, Transformed: {}",
                     convert_to_string(gnss_data.position), convert_to_string(transformed_position));
    }
    
    logger_->info("Adding georeference factor between submap {} and GNSS position {}", 
                 submap->id, convert_to_string(transformed_position));
    
    // Use the symbol 'X' for submaps (as defined in the header)
    gtsam::Key T_key = gtsam::Symbol('T', 0);
    gtsam::Key X_key = X(submap->id);
    
    // Process covariance matrix
    Eigen::Matrix3d prepared_covariance = prepare_covariance(transformed_covariance);
    
    // Create standard Gaussian noise model
    auto noise_model = gtsam::noiseModel::Gaussian::Covariance(prepared_covariance);
    
    // Create our custom factor for position only - this avoids underconstrained situations
    gtsam::NonlinearFactor::shared_ptr factor(new GeoPositionFactor(
        T_key,                 // Key for T transform 
        X_key,                 // Key for X pose
        transformed_position,  // Measured position (with transform applied if available)
        noise_model            // Noise model
    ));
    
    factor_queue_.push_back(std::move(factor));
    
    // Log the factor creation
    logger_->debug("Position factor added with submap position={}, GNSS position={}, covariance_scale={}", 
                  convert_to_string(submap->T_world_origin.translation().eval()),
                  convert_to_string(transformed_position),
                  covariance_scale_);
    
    // Only add rotation constraints in special cases or when we have high confidence
    // This helps avoid underconstrained systems
    // These should be very weak constraints, primarily used for initialization
    if (false) { // Change this condition if you need rotation constraints in special cases
        // Create a weak prior on rotation
        // This is not necessary in most cases and is disabled by default
        // It can be enabled for specific scenarios if needed
        auto rotation_noise = gtsam::noiseModel::Isotropic::Sigma(3, 10.0); // Very high uncertainty
        // Add more rotation constraints here if needed
    }
}

bool Georeference::process_new_data(
    std::deque<GNSSData>& local_gnss_data_queue,
    std::deque<SubMap::ConstPtr>& local_submap_queue)
{
    // Get new raw GNSS data
    const auto new_raw_gnss_data = raw_gnss_data_queue_.get_all_and_clear();
    local_gnss_data_queue.insert(local_gnss_data_queue.end(), 
                               new_raw_gnss_data.begin(), 
                               new_raw_gnss_data.end());

    // Get new submaps
    const auto new_submaps = submap_queue_.get_all_and_clear();
    local_submap_queue.insert(local_submap_queue.end(), 
                            new_submaps.begin(), 
                            new_submaps.end());

    // Sort queues by timestamp if needed
    if (!local_gnss_data_queue.empty()) {
        sort_by_timestamp(local_gnss_data_queue);
    }
    
    if (!local_submap_queue.empty()) {
        std::sort(local_submap_queue.begin(), local_submap_queue.end(),
                 [](const SubMap::ConstPtr& a, const SubMap::ConstPtr& b) {
                     return a->frames.front()->stamp < b->frames.front()->stamp;
                 });
    }
    
    return !new_raw_gnss_data.empty() || !new_submaps.empty();
}

void Georeference::match_submaps_with_gnss(
    std::deque<GNSSData>& local_gnss_data_queue,
    std::deque<SubMap::ConstPtr>& local_submap_queue)
{
    auto submap_iter = local_submap_queue.begin();
    
    while (submap_iter != local_submap_queue.end()) {
        const SubMap::ConstPtr& submap = *submap_iter;
        const auto& frames = submap->frames;
        // Get submap's time range from its first and last frames
        const double submap_start_time = frames.front()->stamp;
        const double submap_end_time = frames.back()->stamp;

        // Sanity check for submap time validity
        if (submap_start_time > submap_end_time) {
            logger_->warn("Submap {} has invalid time window (start={:.3f} > end={:.3f}). Discarding.",
                          submap->id, submap_start_time, submap_end_time);
            submap_iter = local_submap_queue.erase(submap_iter);
            continue;
        }

        if (local_gnss_data_queue.empty()) {
            logger_->debug("No GNSS data available. Waiting for submap {} (t=[{:.3f}-{:.3f}])",
                           submap->id, submap_start_time, submap_end_time);
            break;
        }
        
        if (local_gnss_data_queue.back().timestamp < submap_start_time) {
            // All available GNSS data is older than this submap's start.
            // Need to wait for more GNSS data that might cover this submap.
            logger_->debug("Submap {} (t=[{:.3f}-{:.3f}]) waiting for newer GNSS data. Last GNSS t={:.3f}",
                           submap->id, submap_start_time, submap_end_time, local_gnss_data_queue.back().timestamp);
            break; // Stop processing submaps for this cycle and wait for more GNSS data.
        }

        if (local_gnss_data_queue.front().timestamp > submap_end_time) {
            // All available GNSS data is newer than this submap's end time.
            // This submap is too old and missed its window.
            logger_->warn("Submap {} (t=[{:.3f}-{:.3f}]) is older than all available GNSS data (oldest GNSS t={:.3f}). Discarding.",
                          submap->id, submap_start_time, submap_end_time, local_gnss_data_queue.front().timestamp);
            submap_iter = local_submap_queue.erase(submap_iter);
            continue;
        }

        // Find the best GNSS data for this submap
        auto selected_gnss_points = find_distributed_gnss_points(submap, local_gnss_data_queue);
        if (!selected_gnss_points.empty()) {
            processed_submaps_.push_back(submap);
            submap_gnss_data_map_[submap->id] = selected_gnss_points;
            
            // Calculate and log the average separation between points
            double avg_separation = calculate_average_separation(selected_gnss_points);
            
            logger_->info("Associated submap {} with {} GNSS points, separated by {:.2f}m on average", 
                     submap->id, selected_gnss_points.size(), avg_separation);
            
            submap_iter = local_submap_queue.erase(submap_iter); // Remove processed submap and advance iterator

            // Clean up old GNSS data from local_gnss_data_queue.
            // Remove GNSS data that is older than the start of the earliest remaining submap,
            // or if no submaps are left, older than or equal to the end time of the submap just processed.
            if (!local_gnss_data_queue.empty()) {
                double cleanup_threshold_time;
                bool remove_inclusive = false;

                if (submap_iter != local_submap_queue.end()) { // If there are remaining submaps in local_submap_queue
                    cleanup_threshold_time = (*submap_iter)->frames.front()->stamp;
                } else { // No submaps left in local_submap_queue
                    cleanup_threshold_time = submap_end_time; // Use end time of the submap just processed
                    remove_inclusive = true;
                }
                
                auto& q = local_gnss_data_queue;
                q.erase(std::remove_if(q.begin(), q.end(), [&](const GNSSData& d) {
                    return d.timestamp < cleanup_threshold_time || (remove_inclusive && d.timestamp <= cleanup_threshold_time);
                }), q.end());
            }
        } else {
            logger_->warn("Submap {} (t=[{:.3f}-{:.3f}]): No valid best GNSS candidate found. Discarding submap.",
                          submap->id, submap_start_time, submap_end_time);
            submap_iter = local_submap_queue.erase(submap_iter); // Remove submap and advance iterator
        }
    } // End of while (submap_iter != local_submap_queue.end())
}

void Georeference::add_factors_for_matched_data() {
    // First add factors for all matched data, regardless of initialization status
    for (const auto& submap : processed_submaps_) {
        // Skip if we already processed this submap
        if (processed_submap_ids_.count(submap->id)) {
            continue;
        }
        
        // Get the vector of selected GNSS points for this submap
        auto it = submap_gnss_data_map_.find(submap->id);
        if (it == submap_gnss_data_map_.end()) {
            logger_->warn("No GNSS data found for submap {}", submap->id);
            continue;
        }
        
        const auto& gnss_points = it->second;
        if (gnss_points.empty()) {
            logger_->warn("Empty GNSS data vector for submap {}", submap->id);
            continue;
        }
        
        // Add a factor for each selected GNSS point
        for (const auto& gnss_data : gnss_points) {
            add_georeference_factor(submap, gnss_data);
            logger_->debug("Added georeference factor for submap {} using GNSS data at t={:.3f} with cov_trace={:.4e}",
                         submap->id, gnss_data.timestamp, gnss_data.covariance.trace());
        }
        
        processed_submap_ids_.insert(submap->id);
        logger_->info("Added {} georeference factors for submap {} (avg separation: {:.2f}m)",
                     gnss_points.size(), submap->id, calculate_average_separation(gnss_points));
    }
    
    // Then try to initialize transformation if needed and enough data exists
    if (!transformation_initialized_) {
        if (processed_submap_ids_.size() >= 2) {
            // Check if we have sufficient baseline distance
            if (processed_submaps_.size() >= 2) {
                const auto& first_submap = processed_submaps_.front();
                const auto& last_submap = processed_submaps_.back();
                
                const Eigen::Isometry3d relative_motion = 
                    first_submap->T_world_origin.inverse() * last_submap->T_world_origin;
                
                const double baseline = relative_motion.translation().norm();
                if (baseline > min_baseline_) {
                    transformation_initialized_ = initialize_transformation();
                    if (transformation_initialized_) {
                        logger_->info("Successfully initialized georeference transformation");
                    }
                } else {
                    logger_->debug("Baseline distance ({:.2f}m) too small for initialization, needs > {:.2f}m", 
                                baseline, min_baseline_);
                }
            }
        }
    }
    
    // Remove processed data to save memory if we have accumulated too many
    if (processed_submaps_.size() > MAX_PROCESSED_SUBMAPS) {
        // Get the submaps we're removing
        std::vector<SubMap::ConstPtr> removing(processed_submaps_.begin(), 
                                            processed_submaps_.begin() + CLEANUP_BATCH_SIZE);
        
        // Remove from processed_submaps_
        processed_submaps_.erase(processed_submaps_.begin(), 
                               processed_submaps_.begin() + CLEANUP_BATCH_SIZE);
        
        // Remove corresponding entries from submap_gnss_data_map_
        for (const auto& submap : removing) {
            submap_gnss_data_map_.erase(submap->id);
        }
    }
}

void Georeference::process_data_thread() {
    logger_->info("Starting georeference processing thread");
    
    std::deque<GNSSData> local_gnss_data_queue;      // Stores full GNSSData objects
    std::deque<SubMap::ConstPtr> local_submap_queue;
    
    while (!should_terminate_) {
        try {
            // Process any new data
            bool new_data_processed = process_new_data(local_gnss_data_queue, local_submap_queue);
            
            // Skip further processing if no new data
            if (!new_data_processed) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            // Match submaps with their best GNSS data
            match_submaps_with_gnss(local_gnss_data_queue, local_submap_queue);
            
            // Add factors for matched data
            add_factors_for_matched_data();
            
            // Sleep to avoid high CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } catch (const std::exception& e) {
            logger_->error("Exception in georeference processing thread: {}", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        } catch (...) {
            logger_->error("Unknown exception in georeference processing thread");
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
    
    logger_->info("Georeference processing thread terminated");
}

}  // namespace glim
