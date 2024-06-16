#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <chrono>



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


// -----------------------------------------------------------------------------------------------------------------------------------------------
// RAYTRACER CLASS ----------------------------------------------------------------------------------------------------------------------------------

class Ray {
    public:
        Vector O;
        Vector u;
        explicit Ray(Vector origin, Vector direction) {O = origin;
            u = direction;
        }
};



// Classes from previous project
// -----------------------------------------------------------------------------------------------------------------------------------------------
// INTERSECTION STRUCT ----------------------------------------------------------------------------------------------------------------------------------

struct Intersection {
    
    Vector P;
    Vector N;
    Vector color;
    double t = 0.;
    double refractiveIndex;
    bool isMirror;
    bool intersects = false;

    // Constructor using member initializer list
    Intersection(const Vector& color = Vector(0., 0., 0.), double refractiveIndex = 1.0, bool isMirror = false)
    : color(color), refractiveIndex(refractiveIndex), isMirror(isMirror) {}

};


// -----------------------------------------------------------------------------------------------------------------------------------------------
// GEOMETRY CLASS ----------------------------------------------------------------------------------------------------------------------------------

class Geometry {
public:
    virtual Intersection intersect(const Ray& ray) = 0;
};


// -----------------------------------------------------------------------------------------------------------------------------------------------
// BOUNDINGBOX CLASS ----------------------------------------------------------------------------------------------------------------------------------

class BoundingBox {
public:
    Vector B_min;
    Vector B_max;

    explicit BoundingBox(Vector min = Vector(), Vector max = Vector()) {B_min = min;
        B_max = max;
    }
};


// -----------------------------------------------------------------------------------------------------------------------------------------------
// NODE STRUCT ----------------------------------------------------------------------------------------------------------------------------------

struct Node {
    BoundingBox bbox;
    int startingTriangle;
    int endingTriangle;
    Node* leftChild;
    Node* rightChild;
};

// -----------------------------------------------------------------------------------------------------------------------------------------------
// TRIANGLEINDICES CLASS ----------------------------------------------------------------------------------------------------------------------------------

// from the lecture
class TriangleIndices {
public:
    TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1, bool added = false) : vtxi(vtxi), vtxj(vtxj), vtxk(vtxk), uvi(uvi), uvj(uvj), uvk(uvk), ni(ni), nj(nj), nk(nk), group(group) {
    };
    int vtxi, vtxj, vtxk; // indices within the vertex coordinates array
    int uvi, uvj, uvk;  // indices within the uv coordinates array
    int ni, nj, nk;  // indices within the normals array
    int group;       // face group
};


// -----------------------------------------------------------------------------------------------------------------------------------------------
// TRIANGLE MESH ----------------------------------------------------------------------------------------------------------------------------------

class TriangleMesh : public Geometry {
    double scalingFactor;
    Vector translation;
    Vector color;
    double refractiveIndex;
    bool isMirror;

public:

    std::vector<TriangleIndices> indices;
    std::vector<Vector> vertices;
    std::vector<Vector> normals;
    std::vector<Vector> uvs;
    std::vector<Vector> vertexcolors;
    BoundingBox bbox;
    Node* root;

     ~TriangleMesh() {
        delete root; // Ensure proper cleanup
    }
    TriangleMesh(double scaling_factor, const Vector& translation, const Vector& color = Vector(0., 0., 0.), double refractiveIndex = 1.0, bool isMirror = false)
    : scalingFactor(scaling_factor), translation(translation), color(color), refractiveIndex(refractiveIndex), isMirror(isMirror), root(new Node) {}
    

    void readOBJ(const char* obj) {

        char matfile[255];
        char grp[255];

        FILE* f;
        f = fopen(obj, "r");
        int curGroup = -1;
        while (!feof(f)) {
            char line[255];
            if (!fgets(line, 255, f)) break;

            std::string linetrim(line);
            linetrim.erase(linetrim.find_last_not_of(" \r\t") + 1);
            strcpy(line, linetrim.c_str());

            if (line[0] == 'u' && line[1] == 's') {
                sscanf(line, "usemtl %[^\n]\n", grp);
                curGroup++;
            }

            if (line[0] == 'v' && line[1] == ' ') {
                Vector vec;

                Vector col;
                if (sscanf(line, "v %lf %lf %lf %lf %lf %lf\n", &vec[0], &vec[1], &vec[2], &col[0], &col[1], &col[2]) == 6) {
                    col[0] = std::min(1., std::max(0., col[0]));
                    col[1] = std::min(1., std::max(0., col[1]));
                    col[2] = std::min(1., std::max(0., col[2]));

                    vertices.push_back(vec);
                    vertexcolors.push_back(col);

                } else {
                    sscanf(line, "v %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
                    vertices.push_back(vec);
                }
            }
            if (line[0] == 'v' && line[1] == 'n') {
                Vector vec;
                sscanf(line, "vn %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
                normals.push_back(vec);
            }
            if (line[0] == 'v' && line[1] == 't') {
                Vector vec;
                sscanf(line, "vt %lf %lf\n", &vec[0], &vec[1]);
                uvs.push_back(vec);
            }
            if (line[0] == 'f') {
                TriangleIndices t;
                int i0, i1, i2, i3;
                int j0, j1, j2, j3;
                int k0, k1, k2, k3;
                int nn;
                t.group = curGroup;

                char* consumedline = line + 1;
                int offset;

                nn = sscanf(consumedline, "%u/%u/%u %u/%u/%u %u/%u/%u%n", &i0, &j0, &k0, &i1, &j1, &k1, &i2, &j2, &k2, &offset);
                if (nn == 9) {
                    if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                    if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                    if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                    if (j0 < 0) t.uvi = uvs.size() + j0; else   t.uvi = j0 - 1;
                    if (j1 < 0) t.uvj = uvs.size() + j1; else   t.uvj = j1 - 1;
                    if (j2 < 0) t.uvk = uvs.size() + j2; else   t.uvk = j2 - 1;
                    if (k0 < 0) t.ni = normals.size() + k0; else    t.ni = k0 - 1;
                    if (k1 < 0) t.nj = normals.size() + k1; else    t.nj = k1 - 1;
                    if (k2 < 0) t.nk = normals.size() + k2; else    t.nk = k2 - 1;
                    indices.push_back(t);
                } else {
                    nn = sscanf(consumedline, "%u/%u %u/%u %u/%u%n", &i0, &j0, &i1, &j1, &i2, &j2, &offset);
                    if (nn == 6) {
                        if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                        if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                        if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                        if (j0 < 0) t.uvi = uvs.size() + j0; else   t.uvi = j0 - 1;
                        if (j1 < 0) t.uvj = uvs.size() + j1; else   t.uvj = j1 - 1;
                        if (j2 < 0) t.uvk = uvs.size() + j2; else   t.uvk = j2 - 1;
                        indices.push_back(t);
                    } else {
                        nn = sscanf(consumedline, "%u %u %u%n", &i0, &i1, &i2, &offset);
                        if (nn == 3) {
                            if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                            if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                            if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                            indices.push_back(t);
                        } else {
                            nn = sscanf(consumedline, "%u//%u %u//%u %u//%u%n", &i0, &k0, &i1, &k1, &i2, &k2, &offset);
                            if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                            if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                            if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                            if (k0 < 0) t.ni = normals.size() + k0; else    t.ni = k0 - 1;
                            if (k1 < 0) t.nj = normals.size() + k1; else    t.nj = k1 - 1;
                            if (k2 < 0) t.nk = normals.size() + k2; else    t.nk = k2 - 1;
                            indices.push_back(t);
                        }
                    }
                }

                consumedline = consumedline + offset;

                while (true) {
                    if (consumedline[0] == '\n') break;
                    if (consumedline[0] == '\0') break;
                    nn = sscanf(consumedline, "%u/%u/%u%n", &i3, &j3, &k3, &offset);
                    TriangleIndices t2;
                    t2.group = curGroup;
                    if (nn == 3) {
                        if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                        if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                        if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                        if (j0 < 0) t2.uvi = uvs.size() + j0; else  t2.uvi = j0 - 1;
                        if (j2 < 0) t2.uvj = uvs.size() + j2; else  t2.uvj = j2 - 1;
                        if (j3 < 0) t2.uvk = uvs.size() + j3; else  t2.uvk = j3 - 1;
                        if (k0 < 0) t2.ni = normals.size() + k0; else   t2.ni = k0 - 1;
                        if (k2 < 0) t2.nj = normals.size() + k2; else   t2.nj = k2 - 1;
                        if (k3 < 0) t2.nk = normals.size() + k3; else   t2.nk = k3 - 1;
                        indices.push_back(t2);
                        consumedline = consumedline + offset;
                        i2 = i3;
                        j2 = j3;
                        k2 = k3;
                    } else {
                        nn = sscanf(consumedline, "%u/%u%n", &i3, &j3, &offset);
                        if (nn == 2) {
                            if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                            if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                            if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                            if (j0 < 0) t2.uvi = uvs.size() + j0; else  t2.uvi = j0 - 1;
                            if (j2 < 0) t2.uvj = uvs.size() + j2; else  t2.uvj = j2 - 1;
                            if (j3 < 0) t2.uvk = uvs.size() + j3; else  t2.uvk = j3 - 1;
                            consumedline = consumedline + offset;
                            i2 = i3;
                            j2 = j3;
                            indices.push_back(t2);
                        } else {
                            nn = sscanf(consumedline, "%u//%u%n", &i3, &k3, &offset);
                            if (nn == 2) {
                                if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                                if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                                if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                                if (k0 < 0) t2.ni = normals.size() + k0; else   t2.ni = k0 - 1;
                                if (k2 < 0) t2.nj = normals.size() + k2; else   t2.nj = k2 - 1;
                                if (k3 < 0) t2.nk = normals.size() + k3; else   t2.nk = k3 - 1;
                                consumedline = consumedline + offset;
                                i2 = i3;
                                k2 = k3;
                                indices.push_back(t2);
                            } else {
                                nn = sscanf(consumedline, "%u%n", &i3, &offset);
                                if (nn == 1) {
                                    if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                                    if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                                    if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                                    consumedline = consumedline + offset;
                                    i2 = i3;
                                    indices.push_back(t2);
                                } else {
                                    consumedline = consumedline + 1;
                                }
                            }
                        }
                    }
                }

            }

        }
        fclose(f);

        // this->buildBVH(root, 0, indices.size());
    }
};

//TRYING TO QUICKLY IMPLEMENT SOME NEW STUFF...


class Vertex {
public:
    double x, y;
    bool isBoundary;
    std::vector<int> neighbors;

    Vertex(double x = 0, double y = 0, bool isBoundary = false) 
        : x(x), y(y), isBoundary(isBoundary) {}
};

class Graph {
public:
    std::vector<Vertex> vertices;

    void addVertex(double x, double y, bool isBoundary) {
        vertices.emplace_back(x, y, isBoundary);
    }

    void addEdge(int v1, int v2) {
        vertices[v1].neighbors.push_back(v2);
        vertices[v2].neighbors.push_back(v1);
    }

    void tutteEmbedding() {
        int n = vertices.size();
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd bx = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd by = Eigen::VectorXd::Zero(n);

        for (int i = 0; i < n; ++i) {
            if (vertices[i].isBoundary) {
                A(i, i) = 1;
                bx(i) = vertices[i].x;
                by(i) = vertices[i].y;
            } else {
                A(i, i) = vertices[i].neighbors.size();
                for (int neighbor : vertices[i].neighbors) {
                    A(i, neighbor) = -1;
                }
            }
        }

        Eigen::VectorXd x = A.colPivHouseholderQr().solve(bx);
        Eigen::VectorXd y = A.colPivHouseholderQr().solve(by);

        for (int i = 0; i < n; ++i) {
            vertices[i].x = x(i);
            vertices[i].y = y(i);
        }
    }

    void printVertices() {
        for (const Vertex& v : vertices) {
            std::cout << "Vertex: (" << v.x << ", " << v.y << "), Boundary: " << v.isBoundary << std::endl;
        }
    }
};

int main() {
    Graph graph;

    // Add boundary vertices (example: square boundary)
    graph.addVertex(0, 0, true);
    graph.addVertex(1, 0, true);
    graph.addVertex(1, 1, true);
    graph.addVertex(0, 1, true);

    // Add interior vertices
    graph.addVertex(0.5, 0.5, false);
    graph.addVertex(0.2, 0.7, false);

    // Add edges (example: connecting boundary vertices in a square and some interior edges)
    graph.addEdge(0, 1);
    graph.addEdge(1, 2);
    graph.addEdge(2, 3);
    graph.addEdge(3, 0);
    graph.addEdge(4, 0);
    graph.addEdge(4, 1);
    graph.addEdge(4, 2);
    graph.addEdge(4, 3);
    graph.addEdge(5, 0);
    graph.addEdge(5, 3);

    std::cout << "Before Tutte Embedding:" << std::endl;
    graph.printVertices();

    graph.tutteEmbedding();

    std::cout << "After Tutte Embedding:" << std::endl;
    graph.printVertices();

    return 0;
}