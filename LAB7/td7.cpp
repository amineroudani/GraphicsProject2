#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <chrono>

#include "lbfgs.h"
#include "lbfgs.c"

/**
 * @class Vector
 * @brief A simple vector class for 3D coordinates.
 */
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
    double& operator[](int i) { return coords[i]; }
    
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

//Modifications are made to polygon class
/**
 * @class Polygon
 * @brief Class to represent a polygon with additional functionality.
 */
class Polygon {  
public:
    std::vector<Vector> vertices;

    /**
     * @brief Computes the area of the polygon.
     * @return The area of the polygon.
     */
    double calculateArea() const {
        if (vertices.size() < 3) return 0;
        double area = 0;
        for (size_t idx = 0; idx < vertices.size(); ++idx) {
            const Vector& pointA = vertices[idx];
            const Vector& pointB = vertices[(idx + 1) % vertices.size()];
            area += pointA[0] * pointB[1] - pointA[1] * pointB[0];
        }
        return area / 2;
    }

    /**
     * @brief Integrates the squared distance over the polygon.
     * @param refPoint The point from which distances are measured.
     * @return The integrated squared distance.
     */
    double integrateSquaredDist(const Vector& refPoint) const {
        if (vertices.size() < 3) return 0;
        double totalValue = 0;
        for (size_t idx = 1; idx < vertices.size() - 1; ++idx) {
            Vector tri[3] = {vertices[0], vertices[idx], vertices[idx + 1]};
            double accumulatedValue = 0;

            for (int m = 0; m < 3; ++m) {
                for (int n = m; n < 3; ++n) {
                    accumulatedValue += dot(tri[m] - refPoint, tri[n] - refPoint);
                }
            }
            Vector edge1 = tri[1] - tri[0];
            Vector edge2 = tri[2] - tri[0];
            double triArea = 0.5 * std::abs(edge1[1] * edge2[0] - edge1[0] * edge2[1]);
            totalValue += accumulatedValue / 6 * triArea;
        }
        return totalValue;
    }
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
 * @class PowerDiagram
 * @brief Class to generate and manage a Power diagram.
 */
class PowerDiagram {
public:


    std::vector<Polygon> powerCells;
    std::vector<Vector> inputPoints;
    std::vector<double> weights;

    PowerDiagram() {}
     /**
     * @brief Constructor for PowerDiagram class.
     * @param inputPoints A vector of points to generate the Voronoi diagram.
     * @param weights As defined in the lecture notes
     */
    PowerDiagram(std::vector<Vector>& inputPoints, const std::vector<double>& weights) {
        this->inputPoints = inputPoints;
        this->weights = weights;
    }



    /**
     * @brief Bisects and clips a polygon by the bisector of two points, adjusted for weights.
     * @param polygon The polygon to be bisected and clipped.
     * @param index_0 The index of the first point defining the bisector.
     * @param index_i The index of the second point defining the bisector.
     * @param point1 The first point defining the bisector.
     * @param point2 The second point defining the bisector.
     * @return A polygon representing the clipped polygon.
     */
    Polygon bisectAndClipPolygon(const Polygon& polygon, int index_0, int index_i, const Vector& point1, const Vector& point2) {
        Polygon clippedPolygon;
        Vector midpoint = (point1 + point2) / 2;
        Vector Mprime = midpoint + (weights[index_0] - weights[index_i]) / (2. * (point1 - point2).norm2()) * (point2 - point1);
        clippedPolygon.vertices.reserve(polygon.vertices.size() + 1);

        for (size_t i = 0; i < polygon.vertices.size(); ++i) {
            const Vector& A = (i == 0) ? polygon.vertices[polygon.vertices.size() - 1] : polygon.vertices[i - 1];
            const Vector& B = polygon.vertices[i];
            double t = dot(Mprime - A, point2 - point1) / dot(B - A, point2 - point1);
        
            Vector P = A + t * (B - A);

            if ((B - point1).norm2() - weights[index_0] < (B - point2).norm2() - weights[index_i]) { // B is inside
                if ((A - point1).norm2() - weights[index_0] > (A - point2).norm2() - weights[index_i]) { // A outside
                    clippedPolygon.vertices.push_back(P);
                }
                clippedPolygon.vertices.push_back(B);
            } else if ((A - point1).norm2() - weights[index_0] < (A - point2).norm2() - weights[index_i]) { // A is inside
                clippedPolygon.vertices.push_back(P);
            }
        }
        return clippedPolygon;
    }

    /**
     * @brief Generates the Power Vornoi diagram.
     */
    void generatePowerDiagram() {
        powerCells.resize(inputPoints.size());
        for (int i = 0; i < inputPoints.size(); i++) {
            powerCells[i] = computePowerCell(i);
        }
    }


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
    Polygon computePowerCell(int index) {
        Polygon cell = createBoundingBox();

        for (int i = 0; i < inputPoints.size(); i++) {
            if (i == index) continue;
            cell = bisectAndClipPolygon(cell, index, i, inputPoints[index], inputPoints[i]);
        }
        return cell;
    }

    /**
     * @brief Saves the generated Voronoi diagram to an SVG file.
     * @param filename The name of the SVG file.
     */
    void saveDiagram(std::string filename) {
        save_svg(powerCells, filename, "red");
    }


};

/**
 * @class OptimalTransport
 * @brief Class to perform semi-discrete optimal transport using L-BFGS.
 */
class OptimalTransport {
public:
    OptimalTransport(std::vector<Vector>& pts, const std::vector<double>& lambdas) : inputPoints(pts), lambdas(lambdas) {}

    static lbfgsfloatval_t _evaluate(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step) {
        return reinterpret_cast<OptimalTransport*>(instance)->evaluate(x, g, n, step);
    }
    /**
     * @brief Evaluates the objective function for L-BFGS.
     * @param x The current weights.
     * @param g The gradients.
     * @param n The number of weights.
     * @param step The step size.
     * @return The objective function value.
     */
    lbfgsfloatval_t evaluate(const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step) {
        lbfgsfloatval_t fx = 0.0;

        for (int i = 0; i < n; ++i) {
            solution.weights[i] = x[i];
        }
        solution.generatePowerDiagram();

        double s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < n; ++i) {
            double cellArea = solution.powerCells[i].calculateArea();
            g[i] = -(lambdas[i] - cellArea);
            s1 += solution.powerCells[i].integrateSquaredDist(solution.inputPoints[i]);
            s2 += lambdas[i] * x[i];
            s3 -= x[i] * cellArea;
        }
        fx = s1 + s2 + s3;

        return -fx;
    }

    static int _progress(void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls) {
        return reinterpret_cast<OptimalTransport*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }
    /**
     * @brief Progress callback for L-BFGS.
     * @param x The current weights.
     * @param g The gradients.
     * @param fx The current function value.
     * @param xnorm The norm of x.
     * @param gnorm The norm of g.
     * @param step The step size.
     * @param n The number of weights.
     * @param k The iteration count.
     * @param ls The line search count.
     * @return 0 to continue optimization, non-zero to stop.
     */
    int progress(const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls) {
        for (int i = 0; i < n; ++i) {
            solution.weights[i] = x[i];
        }
        solution.generatePowerDiagram();

        double max_diff = 0;
        for (int i = 0; i < n; ++i) {
            double currentArea = solution.powerCells[i].calculateArea();
            double desiredArea = lambdas[i];
            max_diff = std::max(max_diff, std::abs(currentArea - desiredArea));
        }

        std::cout << "fx: " << fx << " max_diff= " << max_diff << "\t gnorm= " << gnorm << std::endl;

        return 0;
    }

    /**
     * @brief Solves the optimal transport problem.
     */
    void solve() {
        solution.inputPoints = inputPoints;
        solution.weights.resize(inputPoints.size());
        std::fill(solution.weights.begin(), solution.weights.end(), 1.0);

        double fx = 0;
        int ret = lbfgs(inputPoints.size(), &solution.weights[0], &fx, _evaluate, _progress, this, nullptr);

        solution.generatePowerDiagram();
    }

    PowerDiagram solution;
    std::vector<Vector> inputPoints;
    std::vector<double> lambdas;
};

/**
 * @brief Generates a set of random points.
 * @param numPoints The number of points to generate.
 * @return A vector of generated points.
 */

std::vector<Vector> create_random_points(int numPoints) {
    std::vector<Vector> pointSet;
    std::srand(std::time(0));

    for (int i = 0; i < numPoints; ++i) {
        double x = static_cast<double>(std::rand()) / RAND_MAX;
        double y = static_cast<double>(std::rand()) / RAND_MAX;
        pointSet.push_back(Vector(x, y));
    }

    return pointSet;
}

/**
 * @brief Main function to generate and save a Voronoi diagram and solve the optimal transport problem.
 * @return 0 on successful execution.
 */
int main() {
    std::vector<Vector> points(256);
    std::vector<double> lambdas(256);

    for (int i = 0; i < points.size(); i++) {
        points[i][0] = rand() / (double)RAND_MAX;
        points[i][1] = rand() / (double)RAND_MAX;
        points[i][2] = 0;
        lambdas[i] = 1. / points.size();
    }

    OptimalTransport ot(points, lambdas);
    auto start = std::chrono::high_resolution_clock::now();
    ot.solve();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Time: " << duration.count() << " ms" << std::endl;

    ot.solution.saveDiagram("POWER.svg");

    return 0;
}
