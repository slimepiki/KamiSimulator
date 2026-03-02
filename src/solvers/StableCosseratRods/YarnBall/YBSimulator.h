
// Jerry Hsu, jerry.hsu.research@gmail.com, 2025

#pragma once

#include "../../../../extern/glm/glm/common.hpp"
#include "../KittenEngine/includes/modules/Bound.h"
#include <string>
#include <memory>

#include "YarnBall.h"

namespace YarnBall {
using glm::mat4;
using glm::vec3;
using glm::vec4;
using std::shared_ptr;
class Sim {
   public:
    Vertex* verts;
    Kit::Rotor* qs;  // Quaternions orientations
    vec4* qRests;    // Rest angles
    vec3* vels;      // Velocity

    MetaData meta;
    float maxH = 4e-4;           // Largest time step allowed
    Kit::Bound<> currentBounds;  // Current bounding box

    int lastErrorCode = ERROR_NONE;
    int lastWarningCode = ERROR_NONE;

    bool printErrors = true;
    bool renderShaded = false;

   private:
    MetaData* d_meta = nullptr;
    int* d_error = nullptr;
    bool initialized = false;
    Kit::LBVH bvh;

    float lastBVHRebuild = std::numeric_limits<float>::infinity();
    int lastItr = -1;
    size_t stepCounter = 0;

    cudaStream_t stream = nullptr;
    cudaGraphExec_t stepGraph = nullptr;

   public:
    Sim(int numVerts);
    ~Sim();

    inline size_t stepCount() { return stepCounter; }

    // Initializes memory and sets up rest length, angles, and mass
    void configure(float density = 1e-3);

    void setKBend(float k = 2e-6);
    void setKStretch(float k = 1.0);

    void setYoung(float young, float deviation = 1.0);  // young: Young's moludle[GPa]

    // Utils
    // void glueEndpoints(float searchRadius);
    void upload();    // Upload the CPU data to the GPU
    void download();  // Download the GPU data to the CPU
    void zeroVelocities();

    void uploadPosAndVel();
    void downloadPosAndVel();

    // Simulation
    void step(float dt);      // Perform one timestep
    float advance(float dt);  // Advance the simulation by dt using one or more timesteps.

    void printCollisionStats();
    Kitten::LBVH::aabb bounds();

    void exportFiberMesh(std::string path);
    void exportToBCC(std::string path, bool exportAsPolyline = false);
    void exportToOBJ(std::string path);

    // Glue endpoints with a vertex within the search radius
    void glueEndpoints(float searchRadius);

    void SagFree(const std::vector<int>& rootVertices, uint32_t iteration = 1);

   private:
    void uploadMeta();

    void initIterate();
    void endIterate();
    void detectCollisions();
    void iterateCosserat();
    void recomputeStepLimit();
    void checkErrors();

    void rebuildCUDAGraph();
    void SingleSagFree(const std::vector<int>& rootVertices);
};

shared_ptr<Sim> readFromOBJ(std::string path, float targetSegLen, mat4 transform, bool breakUpClosedCurves = false, bool allowResample = true);
shared_ptr<Sim> readFromBCC(std::string path, float targetSegLen, mat4 transform, bool breakUpClosedCurves = false, bool allowResample = true);
shared_ptr<Sim> readFromPoly(std::string path, float targetSegLen, mat4 transform, bool breakUpClosedCurves = false, bool allowResample = true);
shared_ptr<Sim> buildFromJSON(std::string path);
}  // namespace YarnBall