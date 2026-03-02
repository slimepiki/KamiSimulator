#ifndef PNMESH_H_
#define PNMESH_H_

#include <array>
#include <cuda_runtime.h>
#include <chrono>

#include "Kami.h"
#include "utilities/Geometry.h"

// A triangle mesh which has Position and Normal

class PNMesh {
   public:
    struct PNMeshParam {
        uint32_t numVertices;
        uint32_t numFaces;
        glm::mat4 masterTrans;
        glm::mat4 invMasterTrans;
        glm::uvec3 SDFDim;
        Kami::AABB meshAABB;
        string meshPath;
    };

   private:
    static const glm::uvec3 PNMESH_DEFAULT_SDF_SIZE;

    unq_ptr<glm::vec3[]> positions;
    unq_ptr<glm::vec3[]> normals;

    // array of indices of the triangles
    unq_ptr<glm::uvec3[]> triangleIndices;

    PNMeshParam meshParam;
    unq_ptr<Kami::SDF> sdf;
    std::chrono::system_clock::time_point sdfCreated;

    PNMesh();

   public:
    // Please include the extension in filePath.
    PNMesh(string filePath, glm::vec3 pos = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 scale = glm::vec3(1.0f, 1.0f, 1.0f));
    PNMesh(const PNMesh& a);

    void Rotate(float angle, glm::vec3 axis);
    void Translate(glm::vec3 translation);

    // this also can regenerate SDF. The creation time is recorded at sdfCreated.
    void GenerateSDF(glm::uvec3 SDFDim = PNMESH_DEFAULT_SDF_SIZE);

    PNMeshParam& GetPNMeshParamRef() { return meshParam; }
    PNMeshParam GetPNMeshParam() const { return meshParam; }
    unq_ptr<glm::vec3[]>& GetPositionsRef() { return positions; }
    unq_ptr<glm::vec3[]>& GetNormalsRef() { return normals; }
    unq_ptr<glm::uvec3[]>& GetTriangleIndicesRef() { return triangleIndices; }
    unq_ptr<Kami::SDF>& GetSDF() { return sdf; }
    // store the SDF query results
    void StoreSDFQueries(vector<float3>& posVec, vector<float>& distVec, vector<float3>& normVec, sh_ptr<float> sdfCreatedTime = nullptr);
    // pos : a sample point (imput), cp : the nearest mesh's surface point (output), n : normalized normal(output), dist: the distance to surface
    // (output) thickness = mesh's thickness, maxDist : max distance
    bool CollisionTest(const glm::vec3& pos, glm::vec3& cp, glm::vec3& n, float& dist, const float thickness = 0.0, const float maxDist = 0.0);
    double SignedDistanceToMesh(const glm::vec3& pos, const float thickness = 0.0);
    float GetSDFCreatedTime();
};

#endif /* PNMESH_H_ */