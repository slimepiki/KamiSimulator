#ifndef YBCOLLISION_H_
#define YBCOLLISION_H_

#include "../../../../extern/glm/glm/common.hpp"

#include "../KittenEngine/KittenGpuLBVH/lbvh.cuh"
#include "../KittenEngine/includes/modules/Rotor.h"

#define BLOCK_SIZE (32 * 4)
#define THREADS_PER_VERTEX (2)
#define VERTEX_PER_BLOCK (BLOCK_SIZE / THREADS_PER_VERTEX)  // 32*2

namespace YarnBall {
using namespace glm;

enum { ERROR_NONE = 0, ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED, WARNING_SEGMENT_INTERPENETRATION };

// This should really NEVER be exceeded.
constexpr int MAX_COLLISIONS_PER_SEGMENT = 16;

// BCC file header
struct BCCHeader {
    char sign[3];
    unsigned char byteCount;
    char curveType[2];
    char dimensions;
    char upDimension;
    uint64_t curveCount;
    uint64_t totalControlPointCount;
    char fileInfo[40];
};

enum class VertexFlags {
    hasPrev = 1,             // Whether the vertex has a previous vertex
    hasNext = 2,             // Whether the vertex has a next vertex
    hasNextOrientation = 4,  // Whether the segment has a next segment
    fixOrientation = 8,      // Fix the orientation of the segment
    colliding = 16,          // Whether this is colliding (unused)
};

inline bool hasFlag(const uint32_t flags, const VertexFlags flag) { return (flags & (uint32_t)flag) != 0; }

inline uint32_t setFlag(const uint32_t flags, const VertexFlags flag, const bool state) {
    return state ? flags | (uint32_t)flag : flags & ~(uint32_t)flag;
}

// Simulation vertex (aligned to openGL layout)
// This includes everything needed to form the local hessian minus collisions
typedef struct {
    // Linear
    vec3 pos;       // Node position
    float invMass;  // Inverse nodal mass

    float lRest;          // Rest length
    float kStretch;       // Stretching stiffness
    int connectionIndex;  // Index of special one-way connected node, -1 if none. (Used to connect vertices). VertexFlags::hasPrev and
                          // VertexFlags::hasNext also indicate consecutive but for consecutive ones.
    uint32_t flags;       // Flags. see enum VertexFlags
} Vertex;

typedef struct {
    Vertex* d_verts;  // Device vertex array pointer
    Kit::Rotor* d_qs;
    vec4* d_qRests;

    vec3* d_dx;        // Temporary delta position iterants. Stored as deltas for precision.
    vec3* d_vels;      // Velocity
    vec3* d_lastVels;  // Velocity from the last frame

    vec3* d_lastPos;        // Last vertex positions. Temp storage to speed up memory access.
    uint32_t* d_lastFlags;  // Last vertex flags. Temp storage to speed up memory access.
    int* d_lastCID;         // Last cid. Temp storage to speed up memory access.

    int* d_numCols;             // Number of collisions for each segment
    float* d_maxStepSize;       // Max step size for the segment
    int* d_collisions;          // Collisions IDs stored as the other segment index.
    Kit::LBVH::aabb* d_bounds;  // AABBs
    ivec2* d_boundColList;      // Colliding segment AABB IDs.

    vec3 gravity;  // Gravity
    int numItr;    // Number of iterations used per time step

    vec3 worldFloorNormal;  // World floor normal
    float worldFloorPos;    // World floor position

    float h;       // Time step (automatically set)
    float lastH;   // Last time step
    float time;    // Current time
    int numVerts;  // Number of vertices

    float damping;        // Damping forces
    float velocityDecay;  // Velocity decay
    float frictionCoeff;  // Friction coefficient for contacts
    float kCollision;     // Stiffness of the collision
    float kFriction;      // Stiffness of the collision

    float detectionRadius;        // Total detection radius of the yarn (automatically set)
    float scaledDetectionRadius;  // Detection radius scaled by the detectionScaler
    float radius;                 // Yarn radius[m]. Note that this is the minimum radius. The actual radius is r + 0.5 * barrierThickness
    float accelerationRatio;      // Solver acceleration ratio

    float barrierThickness;  // Collision energy barrier thickness. This is the barrier between yarns.
    float detectionScaler;   // The extra room needed for a close by potential collision to be added as a ratio

    float bvhRebuildPeriod;  // The time in between rebuilding the BVH.
    int detectionPeriod;     // The number of steps in between to perform collision detection. -1 to turn off collisions

    float maxSegLen;  // Largest segment length
    float minSegLen;  // Largest segment length
    float density;
    int useStepSizeLimit;   // Whether to use the step size limit
    int useVelocityRadius;  // Whether to add velocity onto the collision radius

    bool enableHairHairColl;

    float ExternalForceMultiplier;

    float windPeak = 0.0f;
    float WindYFreq = 0.0f;
    float WindZFreq = 0.0f;
    float windTimeFreq = 0.0f;
    float windSharpness = 2.0f;

} MetaData;
}  // namespace YarnBall

#endif /* YBCOLLISION_H_ */