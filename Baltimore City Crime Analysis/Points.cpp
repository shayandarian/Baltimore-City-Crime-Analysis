#include <cfloat>

struct Point {
    // MEMBER VARIABLES:
    //  x:          latitude
    //  y:          longitude
    //  minDist:    minimum distance from point to cluster
    //  clusterID:  cluster ID that this point belongs to
    double x;
    double y;
    double minDist = DBL_MAX;
    int clusterID = -1;
    
    // CONSTRUCTORS
    Point() : x(0), y(0), minDist(DBL_MAX), clusterID(-1) {};
    Point(const double x, const double y) : x(x), y(y), minDist(DBL_MAX), clusterID(-1) {};
};

struct Centroid {
    // MEMBER VARIABLES:
    //  x:      latitude
    //  y:      longitude
    //  id:     cluster's ID
    double x;
    double y;
    int id;
    int nPoints;
    
    // CONSTRUCTORS
    Centroid() : x(0), y(0), id(-1), nPoints(0) {};
    Centroid(const double x, const double y) : x(x), y(y), id(-1), nPoints(0) {};

    // MEMBER FUNCTIONS:
    // distance:    calculates distance from given centroid to this point
    double distance(const Point& p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
};