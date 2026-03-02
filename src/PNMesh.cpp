#include "PNMesh.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "../extern/glm/glm/glm.hpp"
#include "../extern/glm/glm/gtx/matrix_decompose.hpp"
#include "utilities/Notification.h"
#include "utilities/Geometry.h"

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/Importer.hpp>

namespace {
const uint ASSIMP_LOAD_FLAGS = (aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices |
                                aiProcess_PopulateArmatureData | aiProcess_GenBoundingBoxes);
}  // namespace

const glm::uvec3 PNMesh::PNMESH_DEFAULT_SDF_SIZE = glm::uvec3(64, 128, 64);

PNMesh::PNMesh(string meshPath, glm::vec3 pos, glm::vec3 scale) {
    // Load with Assimp
    Assimp::Importer import;
    const aiScene* scene = import.ReadFile(meshPath.c_str(), ASSIMP_LOAD_FLAGS);

    auto aiMesh = scene->mMeshes[0];

    auto aiNumVertices = aiMesh->mNumVertices;
    auto aiNumFaces = aiMesh->mNumFaces;

    positions = make_unique<glm::vec3[]>(aiNumVertices);
    normals = make_unique<glm::vec3[]>(aiNumVertices);
    triangleIndices = make_unique<glm::uvec3[]>(aiNumFaces);

    meshParam.numVertices = aiNumVertices;
    meshParam.numFaces = aiNumFaces;

    meshParam.masterTrans = glm::translate(glm::scale(glm::mat4(1.0f), scale), pos);
    meshParam.invMasterTrans = glm::inverse(meshParam.masterTrans);

    meshParam.meshPath = meshPath;

    auto aiAABB = aiMesh->mAABB;
    meshParam.meshAABB.Xmax = aiAABB.mMax.x;
    meshParam.meshAABB.Ymax = aiAABB.mMax.y;
    meshParam.meshAABB.Zmax = aiAABB.mMax.z;
    meshParam.meshAABB.Xmin = aiAABB.mMin.x;
    meshParam.meshAABB.Ymin = aiAABB.mMin.y;
    meshParam.meshAABB.Zmin = aiAABB.mMin.z;

#pragma omp parallel for
    for (uint32_t i = 0; i < aiNumVertices; i++) {
        auto& pos = positions[i];
        auto& normal = normals[i];

        pos.x = aiMesh->mVertices[i].x;
        pos.y = aiMesh->mVertices[i].y;
        pos.z = aiMesh->mVertices[i].z;

        normal.x = aiMesh->mNormals[i].x;
        normal.y = aiMesh->mNormals[i].y;
        normal.z = aiMesh->mNormals[i].z;
    }

#pragma omp parallel for
    for (unsigned int i = 0; i < aiMesh->mNumFaces; i++) {
        aiFace face = aiMesh->mFaces[i];
        auto& tri = triangleIndices[i];
        if (aiMesh->mFaces[i].mNumIndices != 3) {
            Kami::Notification::Warn(
                __func__, "The file " + meshPath + " is the invalid mesh data that has some irregular faces whose number of vertices isn't 3.");
        }
        glm::uvec3 triId = glm::uvec3(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
        tri = triId;
    }
}

PNMesh::PNMesh(const PNMesh& a) {
    meshParam = a.meshParam;
    positions = make_unique<glm::vec3[]>(meshParam.numVertices);
    normals = make_unique<glm::vec3[]>(meshParam.numVertices);
    triangleIndices = make_unique<glm::uvec3[]>(meshParam.numFaces);
    sdfCreated = a.sdfCreated;

    std::copy(a.positions.get(), a.positions.get() + a.meshParam.numVertices, positions.get());
    std::copy(a.normals.get(), a.normals.get() + a.meshParam.numVertices, normals.get());
    std::copy(a.triangleIndices.get(), a.triangleIndices.get() + a.meshParam.numFaces, triangleIndices.get());
    sdf.reset(new Kami::SDF(*a.sdf));
}

void PNMesh::Rotate(float angle, glm::vec3 axis) {
    meshParam.masterTrans = glm::rotate(meshParam.masterTrans, angle, axis);
    meshParam.invMasterTrans = glm::inverse(meshParam.masterTrans);
}

void PNMesh::Translate(glm::vec3 translation) {
    meshParam.masterTrans = glm::translate(meshParam.masterTrans, translation);
    meshParam.invMasterTrans = glm::inverse(meshParam.masterTrans);
}

void PNMesh::GenerateSDF(glm::uvec3 SDFDim) {
    meshParam.SDFDim = SDFDim;

    sdf.reset(new Kami::SDF());
    sdf->Init(positions, triangleIndices, meshParam.numVertices, meshParam.numFaces, SDFDim);
    sdfCreated = std::chrono::system_clock::now();
    // Kami::Notification::Notify("SDF was generated at " + Kami::Notification::GetCurrentTimeStr(sdfCreated) + ".");
}

void PNMesh::StoreSDFQueries(vector<float3>& posVec, vector<float>& distVec, vector<float3>& normVec, sh_ptr<float> sdfCreatedTime) {
    if (!sdf) {
        Kami::Notification::Warn(__func__, "The SDF isn't generated!\n Please call " + Kami::Notification::MakeRedString("GenerateSDF()") +
                                               " if you want to use CollisionTest().");
        return;
    }

    auto gridDim = meshParam.SDFDim;
    auto gridSize = meshParam.meshAABB;
    int totalSize = gridDim.x * gridDim.y * gridDim.z;
    posVec.resize(totalSize);
    distVec.resize(totalSize);
    normVec.resize(totalSize);

    glm::vec3 ds = glm::vec3((gridSize.Xmax - gridSize.Xmin) / gridDim.x, (gridSize.Ymax - gridSize.Ymin) / gridDim.y,
                             (gridSize.Zmax - gridSize.Zmin) / gridDim.z);

    glm::vec3 scale;
    glm::quat rotation;
    glm::vec3 translate = glm::vec3(0, 0, 0);
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(meshParam.masterTrans, scale, rotation, translate, skew, perspective);

    // fill ArraySDF
#pragma omp parallel for
    for (int i = 0; i < totalSize; i++) {
        auto pos = glm::vec3(gridSize.Xmin, gridSize.Ymin, gridSize.Zmin);
        // Idx to 3D idx
        int3 idx3D;
        idx3D.x = i % gridDim.x;
        idx3D.y = (i / gridDim.x) % gridDim.y;
        idx3D.z = i / (gridDim.x * gridDim.y);

        // 3D Idx to position (stored at box vertex)

        pos += ds.x * (idx3D.x) * glm::vec3(1.f, 0.f, 0.f);
        pos += ds.y * (idx3D.y) * glm::vec3(0.f, 1.f, 0.f);
        pos += ds.z * (idx3D.z) * glm::vec3(0.f, 0.f, 1.f);

        glm::vec3 cp, n;
        float dist;
        // Gets SDF
        posVec[i] = make_float3(pos.x, pos.y, pos.z);
        if (GetSDF()->CollisionTest(pos, cp, n, dist)) {
            distVec[i] = dist;
            normVec[i] = make_float3(n.x, n.y, n.z);
            // if (glm::length(n) > 1.0f) {
            //     printf("pos : %f %f %f \tindex : %d %d %d \tdistance : %f\tnormal :  %f %f %f\n", pos.x, pos.y, pos.z, idx3D.x, idx3D.y, idx3D.z,
            //            glm::length(n), n.x, n.y, n.z);
            // }
        } else {
            distVec[i] = GetSDF()->Distance(pos);

            glm::vec3 gridcenter = glm::vec3(gridSize.Xmax, gridSize.Ymax, gridSize.Zmax) + glm::vec3(gridSize.Xmin, gridSize.Ymin, gridSize.Zmin);
            gridcenter /= 2;
            glm::vec3 toCenter = gridcenter - pos;
            float toCentorLength = glm::length(toCenter);
            normVec[i] = make_float3(-toCenter.x / toCentorLength, -toCenter.y / toCentorLength, -toCenter.z / toCentorLength);
        }
    }
    if (sdfCreatedTime) *sdfCreatedTime = GetSDFCreatedTime();
}

bool PNMesh::CollisionTest(const glm::vec3& pos, glm::vec3& cp, glm::vec3& n, float& dist, const float thickness, const float maxDist) {
    if (!sdf) {
        Kami::Notification::Warn(__func__, "The SDF isn't generated!\n Please call " + Kami::Notification::MakeRedString("GenerateSDF()") +
                                               " if you want to use CollisionTest().");
        return false;
    }

    return sdf->CollisionTest(pos, cp, n, dist, thickness, maxDist);
}

double PNMesh::SignedDistanceToMesh(const glm::vec3& pos, const float thickness) {
    if (!sdf) {
        Kami::Notification::Warn(__func__, "The SDF isn't generated!\n Please call " + Kami::Notification::MakeRedString("GenerateSDF()") +
                                               " if you want to use SignedDistanceToMesh().");
        return 0.0;
    }

    return sdf->Distance(pos, thickness);
}

float PNMesh::GetSDFCreatedTime() { return std::chrono::duration_cast<std::chrono::milliseconds>(sdfCreated.time_since_epoch()).count(); }
