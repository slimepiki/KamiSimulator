#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include "../Kami.h"
#include "../../extern/glm/glm/glm.hpp"
#include "../../extern/Discregrid/All"
#include "../../extern/Eigen/Eigen"
#include <array>

#define MAX_BONE_NR 4

namespace Kami {
struct PNVertex {
    glm::vec3 Pos = glm::vec3(0.f);
    glm::vec3 Normal = glm::vec3(0.f);
};

struct PNTVBertex {
    glm::vec3 Pos = glm::vec3(0.f);
    glm::vec3 Normal = glm::vec3(0.f);
    glm::vec2 TexCoord = glm::vec2(0.f);

    glm::vec3 Vel = glm::vec3(0.f);

    int BoneIDs[MAX_BONE_NR];
    float BoneWeights[MAX_BONE_NR];
};

// the structure is from https://github.com/garykac/3d-cubes/blob/master/cube-tex.obj
class CubeObjData {
   public:
    static constexpr glm::uvec2 IDX_SIZE = {8, 3};
    static constexpr glm::uvec2 NORMALS_SIZE = {6, 3};
    static constexpr glm::uvec3 FACES_SIZE = {12, 3, 2};

    // The zero entry means the minimum value of the cube;
    // e.g., {0, 0, 0} means the minimum corner of the box.
    static constexpr int CUBE_IDX[8][3] = {  //
        {0, 0, 0},                           // a
        {0, 1, 0},                           // b
        {1, 1, 0},                           // c
        {1, 0, 0},                           // d
        {0, 0, 1},                           // e
        {0, 1, 1},                           // f
        {1, 1, 1},                           // g
        {1, 0, 1}};

    static constexpr int CUBE_NORMALS[6][3] = {
        {1, 0, 0},   // cghd
        {-1, 0, 0},  // aefb
        {0, 1, 0},   // gcbf
        {0, -1, 0},  // dhea
        {0, 0, 1},   // hgfe
        {0, 0, -1},  // cdab
    };

    //(v,n) means v//n
    static constexpr int CUBE_FACES[12][3][2] = {
        {{3, 1}, {7, 1}, {8, 1}},  // cgh
        {{3, 1}, {8, 1}, {4, 1}},  // chd
        {{1, 2}, {5, 2}, {6, 2}},  // aef
        {{1, 2}, {6, 2}, {2, 2}},  // afb
        {{7, 3}, {3, 3}, {2, 3}},  // aef
        {{7, 3}, {2, 3}, {6, 3}},  // afb
        {{4, 4}, {8, 4}, {5, 4}},  // aef
        {{4, 4}, {5, 4}, {1, 4}},  // afb
        {{8, 5}, {7, 5}, {6, 5}},  // aef
        {{8, 5}, {6, 5}, {5, 5}},  // afb
        {{3, 6}, {4, 6}, {1, 6}},  // aef
        {{3, 6}, {1, 6}, {2, 6}},  // afb
    };
};

struct AABB {
    float Xmin;
    float Xmax;
    float Ymin;
    float Ymax;
    float Zmin;
    float Zmax;

    AABB(float xmin = 0, float xmax = 0, float ymin = 0, float ymax = 0, float zmin = 0, float zmax = 0)
        : Xmin(xmin), Xmax(xmax), Ymin(ymin), Ymax(ymax), Zmin(zmin), Zmax(zmax) {};

    inline void Update(glm::vec3 vec) {
        Xmax = std::max(Xmax, vec.x);
        Xmin = std::min(Xmin, vec.x);
        Ymax = std::max(Ymax, vec.y);
        Ymin = std::min(Ymin, vec.y);
        Zmax = std::max(Zmax, vec.z);
        Zmin = std::min(Zmin, vec.z);
    };
    static inline bool intersection(const AABB& a1, const AABB& a2) {
        if (((a1.Xmax < a2.Xmin) || (a1.Xmin > a2.Xmax))) return false;
        if (((a1.Ymax < a2.Ymin) || (a1.Ymin > a2.Ymax))) return false;
        if (((a1.Zmax < a2.Zmin) || (a1.Zmin > a2.Zmax))) return false;
        return true;
    }
    AABB& operator=(const AABB& aabb) {
        Xmax = aabb.Xmax;
        Xmin = aabb.Xmin;
        Ymax = aabb.Ymax;
        Ymin = aabb.Ymin;
        Zmax = aabb.Zmax;
        Zmin = aabb.Zmin;
        return *this;
    }

    // these does not return reference
    inline Eigen::Vector3f GetMinVertEigenVec3fVal() const { return Eigen::Vector3f(Xmin, Ymin, Zmin); }
    inline Eigen::Vector3f GetMaxVertEigenVec3fVal() const { return Eigen::Vector3f(Xmax, Ymax, Zmax); }
    inline void SetMinVert(Eigen::Vector3f v) {
        Xmin = v[0];
        Ymin = v[1];
        Zmin = v[2];
    }
    inline void SetMaxVert(Eigen::Vector3f v) {
        Xmax = v[0];
        Ymax = v[1];
        Zmax = v[2];
    }
};
class SDF {
   private:
    unq_ptr<Discregrid::CubicLagrangeDiscreteGrid> discregridSDF;

   public:
    SDF() {};
    SDF(const SDF& a);
    void Init(unq_ptr<glm::vec3[]>& vertexPos, unq_ptr<glm::uvec3[]>& triangleIndices, uint32_t numVertices, uint32_t numTriangles,
              glm::uvec3 SDFDim);

    // pos : a sample point (imput), cp : the nearest mesh's surface point (output), n : normalized normal(output), dist: the distance to surface
    // (output) thickness = mesh's thickness, maxDist : max distance
    // this function returns (dist <= maxDist)
    // ( https://github.com/InteractiveComputerGraphics/PositionBasedDynamics/blob/master/Simulation/CubicSDFCollisionDetection.h )
    bool CollisionTest(const glm::vec3& pos, glm::vec3& cp, glm::vec3& n, float& dist, const float thickness = 0.0, const float maxDist = 0.0);
    // ( https://github.com/InteractiveComputerGraphics/PositionBasedDynamics/blob/master/Simulation/CubicSDFCollisionDetection.h )
    double Distance(const glm::vec3& pos, const float thickness = 0.0);
    const std::vector<std::vector<double>>& GetGridNodes() { return discregridSDF->GetMNodes(); }
};
}  // namespace Kami

#endif /* GEOMETRY_H_ */