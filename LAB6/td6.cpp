/**
 * @file voronoi_diagram.cpp
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdio>

class Vector {
public:
    explicit Vector(double x = 0., double y = 0., double z = 0.) {
        coords[0] = x;
        coords[1] = y;
        coords[2] = z;
    }
    Vector& operator+=(const Vector& b) {
        coords[0] += b[0];
        coords[1] += b[1];
        coords[2] += b[2];
        return *this;
    }
    const double& operator[](int i) const { return coords[i]; }
    double operator[](int i) { return coords[i]; }
    
    double norm2() const {
        return coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        coords[0] = coords[0] / n;
        coords[1] = coords[1] / n;
        coords[2] = coords[2] / n;
    }

    bool operator==(const Vector& other) const {
        return coords[0] == other.coords[0] && coords[1] == other.coords[1] && coords[2] == other.coords[2];
    }

private:
    double coords[3];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a * b[0], a * b[1], a * b[2]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0] * b, a[1] * b, a[2] * b);
}
Vector operator*(const Vector& a, const Vector& b) {
    return Vector(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}


// ------------------------------------------------------------------------------------------------------------------------------------------------
// We take the provided code to save .svg files to display your result here: https://pastebin.com/bEYVtqYy as instructed during the lectures.
// saves a static svg file. The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
class Polygon {  
public:
    std::vector<Vector> vertices;
};  

    void save_svg(const std::vector<Polygon> &polygons, std::string filename, std::string fillcol = "none") {
        FILE* f = fopen(filename.c_str(), "w+"); 
        fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
        for (int i=0; i<polygons.size(); i++) {
            fprintf(f, "<g>\n");
            fprintf(f, "<polygon points = \""); 
            for (int j = 0; j < polygons[i].vertices.size(); j++) {
                fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
            }
            fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
            fprintf(f, "</g>\n");
        }
        fprintf(f, "</svg>\n");
        fclose(f);
    }
 
 
// Adds one frame of an animated svg file. frameid is the frame number (between 0 and nbframes-1).
// polygons is a list of polygons, describing the current frame.
// The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
    void save_svg_animated(const std::vector<Polygon> &polygons, std::string filename, int frameid, int nbframes) {
        FILE* f;
        if (frameid == 0) {
            f = fopen(filename.c_str(), "w+");
            fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
            fprintf(f, "<g>\n");
        } else {
            f = fopen(filename.c_str(), "a+");
        }
        fprintf(f, "<g>\n");
        for (int i = 0; i < polygons.size(); i++) {
            fprintf(f, "<polygon points = \""); 
            for (int j = 0; j < polygons[i].vertices.size(); j++) {
                fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000-polygons[i].vertices[j][1] * 1000));
            }
            fprintf(f, "\"\nfill = \"none\" stroke = \"black\"/>\n");
        }
        fprintf(f, "<animate\n");
        fprintf(f, "    id = \"frame%u\"\n", frameid);
        fprintf(f, "    attributeName = \"display\"\n");
        fprintf(f, "    values = \"");
        for (int j = 0; j < nbframes; j++) {
            if (frameid == j) {
                fprintf(f, "inline");
            } else {
                fprintf(f, "none");
            }
            fprintf(f, ";");
        }
        fprintf(f, "none\"\n    keyTimes = \"");
        for (int j = 0; j < nbframes; j++) {
            fprintf(f, "%2.3f", j / (double)(nbframes));
            fprintf(f, ";");
        }
        fprintf(f, "1\"\n   dur = \"5s\"\n");
        fprintf(f, "    begin = \"0s\"\n");
        fprintf(f, "    repeatCount = \"indefinite\"/>\n");
        fprintf(f, "</g>\n");
        if (frameid == nbframes - 1) {
            fprintf(f, "</g>\n");
            fprintf(f, "</svg>\n");
        }
        fclose(f);
    };

// --------------------------------------------------------------------------------

/**
 * @class Voronoi
 * @brief Class to generate and manage a Voronoi diagram from a set of points.
 */
class Voronoi {
public:
    /**
     * @brief Constructor for Voronoi class.
     * @param inputPoints A vector of points to generate the Voronoi diagram.
     */
    Voronoi(std::vector<Vector>& inputPoints) {
        this->inputPoints = inputPoints;
    }

    /**
     * @brief Generates the Voronoi diagram.
     */
    void generateVoronoiDiagram() {
        voronoiCells.resize(inputPoints.size());
        for (int i = 0; i < inputPoints.size(); i++) {
            voronoiCells[i] = computeVoronoiCell(i);
        }
    }

    /**
     * @brief Saves the generated Voronoi diagram to an SVG file.
     * @param filename The name of the SVG file.
     */
    void saveDiagram(std::string filename) {
        save_svg(voronoiCells, filename, "red");
    }

private:
    std::vector<Polygon> voronoiCells;
    std::vector<Vector> inputPoints;

    /**
     * @brief Creates a bounding box for the initial Voronoi cell.
     * @return A polygon representing the bounding box.
     */
    Polygon createBoundingBox() {
        Polygon boundingBox;
        boundingBox.vertices.push_back(Vector(0, 0, 0));
        boundingBox.vertices.push_back(Vector(0, 1, 0));
        boundingBox.vertices.push_back(Vector(1, 1, 0));
        boundingBox.vertices.push_back(Vector(1, 0, 0));
        return boundingBox;
    }

    /**
     * @brief Computes the Voronoi cell for a given point.
     * @param index The index of the point in the input points vector.
     * @return A polygon representing the Voronoi cell.
     */
    Polygon computeVoronoiCell(int index) {
        Polygon cell = createBoundingBox();

        for (int i = 0; i < inputPoints.size(); i++) {
            if (i == index) continue;
            cell = bisectAndClipPolygon(cell, inputPoints[index], inputPoints[i]);
        }
        return cell;
    }

    /**
     * @brief Bisects and clips a polygon by the bisector of two points.
     * @param polygon The polygon to be bisected and clipped.
     * @param point1 The first point defining the bisector.
     * @param point2 The second point defining the bisector.
     * @return A polygon representing the clipped polygon.
     */
    Polygon bisectAndClipPolygon(const Polygon& polygon, const Vector& point1, const Vector& point2) {
        Polygon clippedPolygon;
        Vector midpoint = (point1 + point2) / 2;
        Vector direction = point2 - point1;

        for (int i = 0; i < polygon.vertices.size(); i++) {
            const Vector& startVertex = polygon.vertices[i];
            const Vector& endVertex = polygon.vertices[(i + 1) % polygon.vertices.size()];

            bool startInside = isPointInsideVoronoiCell(startVertex, point1, point2);
            bool endInside = isPointInsideVoronoiCell(endVertex, point1, point2);

            if (startInside != endInside) {
                double t = dot(midpoint - startVertex, direction) / dot(endVertex - startVertex, direction);
                Vector intersectionPoint = startVertex + t * (endVertex - startVertex);
                clippedPolygon.vertices.push_back(intersectionPoint);
            }

            if (endInside) {
                clippedPolygon.vertices.push_back(endVertex);
            }
        }

        return clippedPolygon;
    }

    /**
     * @brief Checks if a point is inside a Voronoi cell.
     * @param point The point to check.
     * @param cellCenter The center of the Voronoi cell.
     * @param otherPoint The other point defining the Voronoi cell.
     * @return True if the point is inside the Voronoi cell, false otherwise.
     */
    bool isPointInsideVoronoiCell(const Vector& point, const Vector& cellCenter, const Vector& otherPoint) {
        return (point - cellCenter).norm2() < (point - otherPoint).norm2();
    }
};

/**
 * @brief Generates a set of random points.
 * @param numPoints The number of points to generate.
 * @return A vector of generated points.
 */
std::vector<Vector> createRandomPoints(int numPoints) {
    std::vector<Vector> pointSet;
    std::srand(std::time(0)); // Seed the random number generator

    for (int i = 0; i < numPoints; ++i) {
        double x = static_cast<double>(std::rand()) / RAND_MAX;
        double y = static_cast<double>(std::rand()) / RAND_MAX;
        pointSet.push_back(Vector(x, y));
    }

    return pointSet;
}

/**
 * @brief Main function to generate and save a Voronoi diagram.
 * @return 0 on successful execution.
 */
int main() {
    int numberOfPoints = 60; // Number of random points to generate
    std::vector<Vector> randomSites = createRandomPoints(numberOfPoints);
    Voronoi voronoiDiagram(randomSites);
    voronoiDiagram.generateVoronoiDiagram();
    voronoiDiagram.saveDiagram("VORNOI.svg");
    return 0;
}
