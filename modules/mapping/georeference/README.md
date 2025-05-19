# Georeference Module for GLIM

## Overview

The Georeference module integrates GNSS (Global Navigation Satellite System) data with GLIM's mapping framework, enabling globally-referenced maps. This module establishes a transformation between the local coordinate system used by SLAM/odometry and a global reference frame (typically ECEF - Earth-Centered, Earth-Fixed coordinates).

## Input Types

The georeference module can accept different types of GNSS input:

- **NavSatFix**: GPS latitude/longitude/altitude with covariance (automatically converted to ECEF)
- **Odometry**: Position with covariance information (such as from navsat_transform)
- **PoseStamped**: Simple pose

## Configuration

Configuration parameters are defined in `config_georeference.json`:

### Basic Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_topic` | string | "/georeference_input" | Topic to subscribe for georeference data |
| `input_type` | string | "PoseStamped" | Type of input data ("NavSatFix", "Odometry", or "PoseStamped") |
| `min_baseline` | double | 5.0 | Minimum baseline distance (m) required for transformation initialization |

### Grid-based Point Selection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_cell_size` | double | 2.0 | Size of each grid cell for spatial distribution (m) |

### Covariance Handling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `covariance_scale` | double | 10.0 | Scaling factor for GNSS covariance matrices |
| `min_covariance_eigenvalue` | double | 1e-3 | Minimum eigenvalue for covariance matrix regularization |
| `position_prediction_window` | double | 5.0 | Time window for position prediction (s) |

### Frame Transformation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lidar_gps_translation` | Vector3d | [0,0,0] | Translation vector from LiDAR to GPS (m) |
| `lidar_gps_rotation_rpy` | Vector3d | [0,0,0] | Rotation from LiDAR to GPS in roll-pitch-yaw (rad) |

## How It Works

The georeference module:

1. **Collects GNSS Data**: Subscribes to GNSS topics and processes incoming measurements
2. **Selects Points**: Uses a grid-based sampling method to select spatially distributed GNSS points  
3. **Associates with Submaps**: Matches GNSS points with corresponding submaps based on timestamps
4. **Creates Factors**: Generates georeference factors connecting submaps to global positions
5. **Optimizes Transformation**: Continuously refines the transformation between coordinate systems 
6. **Updates Submap Poses**: Applies the global transformation to all submaps

## TODO

- Make merging multiple mapping sessions possible preserving global coordinates
- Grab lidar / gps transform from a TF msg
- More sensible handling of non-NavSatFix msgs
