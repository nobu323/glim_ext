#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace glim {

/**
 * @brief Convert ECEF coordinates to WGS84 (latitude, longitude, altitude)
 * @param ecef ECEF coordinates (x, y, z)
 * @return WGS84 coordinates (latitude, longitude, altitude) in degrees and meters
 */
Eigen::Vector3d ecef_to_wgs84(const Eigen::Vector3d& ecef);

/**
 * @brief Convert WGS84 coordinates to ECEF
 * @param lat Latitude in degrees
 * @param lon Longitude in degrees
 * @param alt Altitude in meters
 * @return ECEF coordinates (x, y, z) in meters
 */
Eigen::Vector3d wgs84_to_ecef(double lat, double lon, double alt);

/**
 * @brief Calculate transformation from ECEF to local North-West-Up frame
 * @param ecef ECEF coordinates of the origin
 * @param radius Earth radius (default = 6378137 meters for WGS84)
 * @return Transformation matrix from ECEF to local NWZ frame
 */
Eigen::Isometry3d calc_T_ecef_nwz(const Eigen::Vector3d& ecef, double radius = 6378137);

/**
 * @brief Calculate great-circle distance between two points on Earth's surface
 * @param latlon1 First point coordinates (latitude, longitude) in radians
 * @param latlon2 Second point coordinates (latitude, longitude) in radians
 * @return Distance in meters
 */
double haversine(const Eigen::Vector2d& latlon1, const Eigen::Vector2d& latlon2);

}  // namespace glim