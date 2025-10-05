#include <glim_ext/geodetic.hpp>
#include <cmath>

namespace glim {

// Earth parameters (WGS84 ellipsoid)
namespace {
    constexpr double kSemimajorAxis = 6378137.0;                // Earth's semi-major axis (a) in meters
    constexpr double kSemiminorAxis = 6356752.3142;             // Earth's semi-minor axis (b) in meters
    constexpr double kFirstEccentricitySquared = 6.69437999014e-3;  // e^2
    constexpr double kSecondEccentricitySquared = 6.73949674228e-3; // e'^2
    constexpr double kFlattening = 1.0 / 298.257223563;         // Flattening (f)
}

Eigen::Vector3d ecef_to_wgs84(const Eigen::Vector3d& ecef) {
    // Convert ECEF coordinates to geodetic coordinates using Zhu's method
    // J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates
    // to geodetic coordinates," IEEE Transactions on Aerospace and
    // Electronic Systems, vol. 30, pp. 957-961, 1994.

    const double x = ecef.x();
    const double y = ecef.y();
    const double z = ecef.z();

    const double r = std::sqrt(x * x + y * y);
    const double Esq = kSemimajorAxis * kSemimajorAxis - kSemiminorAxis * kSemiminorAxis;
    const double F = 54.0 * kSemiminorAxis * kSemiminorAxis * z * z;
    const double G = r * r + (1.0 - kFirstEccentricitySquared) * z * z - kFirstEccentricitySquared * Esq;
    const double C = (kFirstEccentricitySquared * kFirstEccentricitySquared * F * r * r) / std::pow(G, 3);
    const double S = std::cbrt(1.0 + C + std::sqrt(C * C + 2.0 * C));
    const double P = F / (3.0 * std::pow((S + 1.0 / S + 1.0), 2) * G * G);
    const double Q = std::sqrt(1.0 + 2.0 * kFirstEccentricitySquared * kFirstEccentricitySquared * P);
    
    const double r_0 = -(P * kFirstEccentricitySquared * r) / (1.0 + Q) +
                       std::sqrt(0.5 * kSemimajorAxis * kSemimajorAxis * (1.0 + 1.0 / Q) - 
                                P * (1.0 - kFirstEccentricitySquared) * z * z / (Q * (1.0 + Q)) - 
                                0.5 * P * r * r);
                                
    const double U = std::sqrt(std::pow((r - kFirstEccentricitySquared * r_0), 2) + z * z);
    const double V = std::sqrt(std::pow((r - kFirstEccentricitySquared * r_0), 2) + (1.0 - kFirstEccentricitySquared) * z * z);
    const double Z_0 = kSemiminorAxis * kSemiminorAxis * z / (kSemimajorAxis * V);

    const double alt = U * (1.0 - kSemiminorAxis * kSemiminorAxis / (kSemimajorAxis * V));
    const double lat = std::atan((z + kSecondEccentricitySquared * Z_0) / r) * 180.0 / M_PI;
    const double lon = std::atan2(y, x) * 180.0 / M_PI;

    return {lat, lon, alt};
}

Eigen::Vector3d wgs84_to_ecef(double lat, double lon, double alt) {
    const double lat_rad = lat * M_PI / 180.0;
    const double lon_rad = lon * M_PI / 180.0;
    
    const double sin_lat = std::sin(lat_rad);
    const double cos_lat = std::cos(lat_rad);
    const double sin_lon = std::sin(lon_rad);
    const double cos_lon = std::cos(lon_rad);
    
    const double xi = std::sqrt(1.0 - kFirstEccentricitySquared * sin_lat * sin_lat);
    const double N = kSemimajorAxis / xi;  // Radius of curvature in the prime vertical
    
    const double x = (N + alt) * cos_lat * cos_lon;
    const double y = (N + alt) * cos_lat * sin_lon;
    const double z = (N * (1.0 - kFirstEccentricitySquared) + alt) * sin_lat;

    return {x, y, z};
}

Eigen::Isometry3d calc_T_ecef_nwz(const Eigen::Vector3d& ecef, double radius) {
    // Calculate the local North-West-Up frame at the given ECEF position
    const Eigen::Vector3d up = ecef.normalized();
    const Eigen::Vector3d to_north = (Eigen::Vector3d::UnitZ() * radius - ecef).normalized();
    const Eigen::Vector3d north = (to_north - to_north.dot(up) * up).normalized();
    const Eigen::Vector3d west = up.cross(north);

    Eigen::Isometry3d T_ecef_nwz = Eigen::Isometry3d::Identity();
    T_ecef_nwz.linear().col(0) = north;
    T_ecef_nwz.linear().col(1) = west;
    T_ecef_nwz.linear().col(2) = up;
    T_ecef_nwz.translation() = ecef;

    return T_ecef_nwz;
}

double haversine(const Eigen::Vector2d& latlon1, const Eigen::Vector2d& latlon2) {
    const double lat1 = latlon1[0];
    const double lon1 = latlon1[1];
    const double lat2 = latlon2[0];
    const double lon2 = latlon2[1];

    const double dlat = lat2 - lat1;
    const double dlon = lon2 - lon1;

    const double a = std::pow(std::sin(dlat / 2.0), 2) + 
                     std::cos(lat1) * std::cos(lat2) * std::pow(std::sin(dlon / 2.0), 2);
    const double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));

    return kSemimajorAxis * c;
}

}  // namespace glim