
// Jerry Hsu, jerry.hsu.research@gmail.com, 2025

#include "YBSimulator.h"
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../KittenEngine/includes/modules/Mesh.h"
#include "../KittenEngine/includes/modules/Common.h"

#define KTR Kitten::Rotor

#define checkCudaErrors(ans)                  \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

[[maybe_unused]]
inline void CkGPUMem() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("%zu KB free of total %zu KB\n", free / 1024, total / 1024);
}

namespace YarnBall {
extern __global__ void cosseratItr(MetaData* data);
extern __global__ void PBDQuat(MetaData* data);
extern __global__ void quaternionLambdaItr(MetaData* data);
extern __global__ void initItr(MetaData* data);
extern __global__ void endItr(MetaData* data);

__host__ void initStepLimit(MetaData* data, int numVerts);
__global__ void initStepLimitKernel(MetaData* data);
template <bool USE_VEL_RADIUS>
__global__ void buildAABBs(MetaData* data, int* errorReturn);
extern template __global__ void buildAABBs<true>(MetaData* data, int* errorReturn);
extern template __global__ void buildAABBs<false>(MetaData* data, int* errorReturn);

template <bool USE_VEL_RADIUS>
extern __global__ void buildCollisionList(MetaData* data, int maxCols, int* errorReturn);
extern template __global__ void buildCollisionList<true>(MetaData* data, int maxCols, int* errorReturn);
extern template __global__ void buildCollisionList<false>(MetaData* data, int maxCols, int* errorReturn);

template <bool LIMIT, bool USE_VEL_RADIUS>
extern __global__ void recomputeStepLimitKernel(MetaData* data);
extern template __global__ void recomputeStepLimitKernel<true, true>(MetaData* data);
extern template __global__ void recomputeStepLimitKernel<true, false>(MetaData* data);
extern template __global__ void recomputeStepLimitKernel<false, true>(MetaData* data);
extern template __global__ void recomputeStepLimitKernel<false, false>(MetaData* data);

const static double r2d = 57.2957795130823208767981548141051703324;
const static double myPi = 3.141592653589793238462643;
const static double my2Pi = 6.2831853071795864769252867665590057683943;
const static double myPi2 = 1.5707963267948966192313216916397;

double my_fmod(double x, double y) {
    double ans;

    long double lx = x;
    long double ly = y;

    if (ly != 0) {
        ans = lx - (long long int)(lx / ly) * ly;
    } else {
        ans = 0;
    }

    return ans;
}

double fitAngle(double angle, bool test = false) {
    if (test) printf("angle: %.10f -> ", angle * r2d);
    angle = my_fmod(my_fmod(angle + myPi, my2Pi) + my2Pi, my2Pi) - myPi;
    if (test) printf("%.10f\n\n", angle * r2d);
    return angle;
}

Sim::Sim(int numVerts) {
    if (numVerts < 3) throw std::runtime_error("Too little vertices");
    meta.numVerts = numVerts;
    meta.gravity = vec3(0, -9.8, 0);
    meta.h = maxH;
    meta.velocityDecay = 0.2;
    meta.damping = 1e-6;
    meta.time = 0.f;

    meta.radius = 1e-4;
    meta.barrierThickness = 8e-4;
    meta.accelerationRatio = 1;

    meta.kCollision = 1e-5;
    meta.detectionScaler = 1.2f;
    meta.frictionCoeff = 0.1f;
    meta.kFriction = 5.f;
    meta.ExternalForceMultiplier = 1;

    meta.detectionPeriod = 1;
    meta.useStepSizeLimit = false;
    meta.useVelocityRadius = true;
    meta.bvhRebuildPeriod = 1 / 10.f;  // Really only need to rebuild every 100ms
    meta.numItr = 8;

    meta.enableHairHairColl = false;
    meta.windPeak = 0.0f;
    meta.WindYFreq = 0.0;
    meta.WindZFreq = 0.0;
    meta.windTimeFreq = 0.0;
    meta.windSharpness = 1.0;

    // Initialize vertices
    verts = new Vertex[numVerts];
    vels = new vec3[numVerts];
    qs = new KTR[numVerts];
    qRests = new vec4[numVerts];
    for (size_t i = 0; i < numVerts; i++) {
        verts[i].invMass = verts[i].lRest = 1;
        vels[i] = vec3(0);
        qRests[i] = (KTR::identity()).v;
        verts[i].kStretch = 100.f;
        verts[i].connectionIndex = -1;
        verts[i].flags = (uint32_t)VertexFlags::hasNext;
    }
    verts[numVerts - 1].flags = 0;
}

Sim::~Sim() {
    delete[] verts;
    delete[] vels;
    delete[] qs;
    delete[] qRests;
    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    if (d_meta) {
        cudaFree(meta.d_dx);
        cudaFree(meta.d_vels);
        cudaFree(meta.d_qRests);

        cudaFree(meta.d_lastVels);
        cudaFree(meta.d_lastPos);
        cudaFree(meta.d_lastFlags);
        cudaFree(meta.d_lastCID);
        cudaFree(meta.d_numCols);
        cudaFree(meta.d_maxStepSize);
        cudaFree(meta.d_collisions);
        cudaFree(meta.d_bounds);
        cudaFree(meta.d_boundColList);
        cudaFree(d_meta);
    }
    if (d_error) cudaFree(d_error);
    if (stepGraph) cudaGraphExecDestroy(stepGraph);
}

Kitten::LBVH::aabb Sim::bounds() { return bvh.bounds(); }

void Sim::configure(float density) {
    const int numVerts = meta.numVerts;

    meta.density = density;
    meta.maxSegLen = 0;
    meta.minSegLen = FLT_MAX;

    auto lastQ = KTR::identity();
    // Init mass and orientation
    for (int i = 0; i < numVerts; i++) {
        auto& v = verts[i];

        // Fix flags
        if (i < numVerts - 1) {
            bool hasNext = hasFlag(v.flags, VertexFlags::hasNext);
            verts[i + 1].flags = setFlag(verts[i + 1].flags, VertexFlags::hasPrev, hasNext);

            bool hasNextNext = hasNext && hasFlag(verts[i + 1].flags, VertexFlags::hasNext);
            v.flags = setFlag(v.flags, VertexFlags::hasNextOrientation, hasNextNext);

            // If the segment doesnt exist, then we fix the rotation
            if (!hasNext) v.flags |= (uint32_t)VertexFlags::fixOrientation;
        }

        if (!hasFlag(v.flags, VertexFlags::hasPrev) && !hasFlag(v.flags, VertexFlags::hasNext))
            throw std::runtime_error("Dangling segment. Yarns must be atleast 2 segments long");

        v.lRest = 1.f / numVerts;

        float mass = 0;
        if (v.flags & (uint32_t)VertexFlags::hasPrev) mass += verts[i - 1].lRest;

        if (v.flags & (uint32_t)VertexFlags::hasNext) {
            auto& v1 = verts[i + 1];
            vec3 seg0 = v1.pos - v.pos;
            v.lRest = length(seg0);
            if (v.lRest == 0 || !glm::isfinite(v.lRest)) throw std::runtime_error("0 length segment");

            auto qq = KTR::fromTo(glm::vec3(1, 0, 0), normalize(seg0));

            //// regacy
            //// Init orientation based on minimizing twist t with
            //// min |q1 t - q0|^2 = min |t - q1^-1 q0|^2
            //// i.e. t is just the normalized x and w components.
            // auto t = qq.inverse() * lastQ;
            // t = normalize(vec4(t.x, 0, 0, t.w));

            lastQ = qq;

            mass += v.lRest;

            meta.maxSegLen = max(meta.maxSegLen, v.lRest);
            meta.minSegLen = min(meta.minSegLen, v.lRest);
        }
        qs[i] = lastQ;
        // if (i < 100) {
        //     printf("i = %d\n", i);
        //     printf("i(%.15f, %.15f, %.15f)\n", verts[i].pos.x, verts[i].pos.y, verts[i].pos.z);
        //     printf("i+1(%.15f, %.15f, %.15f)\n", verts[i + 1].pos.x, verts[i + 1].pos.y, verts[i + 1].pos.z);
        //     auto i1mi = verts[i + 1].pos - verts[i].pos;
        //     printf("i+1-i(%.15f, %.15f, %.15f)\n\n", i1mi.x, i1mi.y, i1mi.z);
        // }

        // if (!i) printf("SCR mass: %f, ours mass: %.15f\n\n", mass * 0.5f * density, mass * (meta.radius) * (meta.radius) * density * 0.5f);
        // if (i == 1) printf("radius: %.10f, density: %.10f, length: %.10f, ", meta.radius, density, mass);

        // mass = (l_(i-1) + l_i) * r ^ 2 * pi * density / 2
        // [kg] = ([m])* ([m]) ^ 2 * [kg/m^3]/2
        mass *= (meta.radius) * (meta.radius) * myPi * density * 0.5f;

        //  mass *= 0.5f * density; // original code

        if (mass != 0)
            v.invMass *= 1 / mass;
        else
            v.invMass = 0;
    }

    // Init rest orientation
    // for (int i = 0; i < numVerts - 1; i++) qRests[i] = length(qRests[i]) * (qs[i].inverse() * qs[i + 1]).v;
    for (int i = 0; i < numVerts - 1; i++) qRests[i] = glm::normalize((qs[i].inverse() * qs[i + 1]).v);

    // Init meta
    cudaMalloc(&d_meta, sizeof(MetaData));

    cudaMalloc(&d_error, 2 * sizeof(int));
    cudaMemset(d_error, 0, 2 * sizeof(int));

    cudaMalloc(&meta.d_dx, sizeof(vec3) * numVerts);

    cudaMalloc(&meta.d_verts, sizeof(Vertex) * numVerts);
    cudaMalloc(&meta.d_qs, sizeof(KTR) * numVerts);
    cudaMalloc(&meta.d_lastVels, sizeof(vec3) * numVerts);
    cudaMemset(meta.d_lastVels, 0, sizeof(vec3) * numVerts);
    cudaMalloc(&meta.d_vels, sizeof(vec3) * numVerts);
    cudaMalloc(&meta.d_qRests, sizeof(vec4) * numVerts);
    cudaMalloc(&meta.d_lastPos, sizeof(vec3) * numVerts);
    cudaMalloc(&meta.d_lastCID, sizeof(int) * numVerts);
    cudaMalloc(&meta.d_lastFlags, sizeof(int) * numVerts);

    cudaMalloc(&meta.d_maxStepSize, sizeof(float) * numVerts);
    cudaMalloc(&meta.d_numCols, sizeof(int) * numVerts);
    cudaMemset(meta.d_numCols, 0, sizeof(int) * meta.numVerts);
    cudaMalloc(&meta.d_collisions, sizeof(int) * numVerts * MAX_COLLISIONS_PER_SEGMENT);
    cudaMalloc(&meta.d_bounds, sizeof(Kit::LBVH::aabb) * numVerts);
    cudaMalloc(&meta.d_boundColList, sizeof(int) * numVerts * MAX_COLLISIONS_PER_SEGMENT);

    cudaDeviceSynchronize();
    cudaStreamCreate(&stream);
    uploadMeta();
    upload();
    initStepLimit(d_meta, meta.numVerts);
    checkCudaErrors(cudaGetLastError());
}

void Sim::setKStretch(float kStretch) {
    if (!d_meta) throw std::runtime_error("No rest length. Must call configure()");

    // Multiplied by rest length to make energy density consistent.
    // Each segment has l * E energy, where E = C.k.C
    // The l is moved into the kStretch
    for (int i = 0; i < meta.numVerts; i++) verts[i].kStretch = kStretch * verts[i].lRest;
    // printf("verts[0].kStretch: %.15f\n\n", verts[0].kStretch);
}

void Sim::setKBend(float kBend) {
    if (!d_meta) throw std::runtime_error("No rest length. Must call configure()");

    // Scaled by the 4 below
    kBend *= 4;

    // Divded by rest length to make energy density consistent.
    // Each segment has l * E energy, where E = C.k.C
    // The l is moved into the kBend, but we also cheated because the darboux vectors
    // in C should have been scaled by 2/l. So in total we end up dividing once.
    for (int i = 0; i < meta.numVerts; i++) qRests[i] = (kBend / verts[i].lRest) * normalize((vec4)qRests[i]);
    // printf("verts[0].kBend: %.15f\n\n", kBend / verts[0].lRest);
}

void Sim::setYoung(float young, float deviation) {
#pragma omp parallel for
    float mmrad = meta.radius * 1000;
    for (int i = 0; i < meta.numVerts; i++) {
        float len = verts[i].lRest;
        float ss = young * myPi * mmrad * mmrad;
        float bt = ss * deviation * mmrad * mmrad * 0.25 / (len * len);

        //        float len = verts[i].lRest;
        // float ss = young * myPi * meta.radius * meta.radius / len;
        // float bt = ss * deviation * meta.radius * meta.radius * 0.25;

        // if (!i) {
        //     printf("young: %f, mmrad: %f\n", young, mmrad);
        //     printf("len: %f,ss: %f, bt: %f\n\n", len, ss, bt);
        // }

        verts[i].kStretch = ss;
        qRests[i] *= bt;
    }
}

void Sim::uploadMeta() {
    meta.detectionRadius = meta.radius + 0.5f * meta.barrierThickness;
    meta.scaledDetectionRadius = meta.detectionRadius * meta.detectionScaler;

    if (meta.detectionScaler <= 1) throw std::runtime_error("Detection scaler must be greater than 1.");

    if (2 * meta.minSegLen < 2 * meta.radius + meta.barrierThickness) {
        printf("minSegLen: %f, radius %f, barrierThickness %f \n", meta.minSegLen, meta.radius, meta.barrierThickness);
        printf("left: %f, right %f\n", 2 * meta.minSegLen, 2 * meta.radius + meta.barrierThickness);
        throw std::runtime_error("Use thinner yarn or use longer segments. (2 * min_seg_length must be > 2 * radius + barrier_thickness)");
    }

    cudaMemcpyAsync(d_meta, &meta, sizeof(MetaData), cudaMemcpyHostToDevice, stream);
}

__global__ void copyTempData(Vertex* verts, uint32_t* lastFlags, int* lastCID, int numVerts) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numVerts) return;
    auto v = verts[tid];
    lastFlags[tid] = v.flags;
    lastCID[tid] = v.connectionIndex;
}

void Sim::upload() {
    // printf(("invmass: %.10f, kstretch:, %f, kbend: %f\n"), verts[10000].invMass, verts[10000].kStretch, glm::length(qRests[10000]));
    cudaMemcpyAsync(meta.d_verts, verts, sizeof(Vertex) * meta.numVerts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(meta.d_vels, vels, sizeof(vec3) * meta.numVerts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(meta.d_qs, qs, sizeof(KTR) * meta.numVerts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(meta.d_qRests, qRests, sizeof(vec4) * meta.numVerts, cudaMemcpyHostToDevice, stream);
    copyTempData<<<(meta.numVerts + 511) / 512, 512, 0, stream>>>(meta.d_verts, meta.d_lastFlags, meta.d_lastCID, meta.numVerts);
    cudaStreamSynchronize(stream);
}

void Sim::download() {
    cudaMemcpyAsync(verts, meta.d_verts, sizeof(Vertex) * meta.numVerts, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(vels, meta.d_vels, sizeof(vec3) * meta.numVerts, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(qs, meta.d_qs, sizeof(KTR) * meta.numVerts, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

__global__ void zeroVels(vec3* vels, vec3* lastVels, int numVerts) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numVerts) return;

    vels[tid] = vec3(0);
    lastVels[tid] = vec3(0);
}

void Sim::zeroVelocities() {
    zeroVels<<<(meta.numVerts + 1023) / 1024, 1024, 0, stream>>>(meta.d_vels, meta.d_lastVels, meta.numVerts);
    checkCudaErrors(cudaGetLastError());
}

void Sim::uploadPosAndVel() {
    cudaMemcpyAsync(meta.d_verts, verts, sizeof(Vertex) * meta.numVerts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(meta.d_vels, vels, sizeof(vec3) * meta.numVerts, cudaMemcpyHostToDevice, stream);
    copyTempData<<<(meta.numVerts + 511) / 512, 512, 0, stream>>>(meta.d_verts, meta.d_lastFlags, meta.d_lastCID, meta.numVerts);
    cudaStreamSynchronize(stream);
}

void Sim::downloadPosAndVel() {
    cudaMemcpyAsync(verts, meta.d_verts, sizeof(Vertex) * meta.numVerts, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(vels, meta.d_vels, sizeof(vec3) * meta.numVerts, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

void Sim::checkErrors() {
    checkCudaErrors(cudaGetLastError());

    int error[2];
    cudaMemcpyAsync(error, d_error, 2 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (error[0] == ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED) {
        fprintf(stderr, "ERROR: MAX_COLLISIONS_PER_SEGMENT exceeded. Current simulation state may be corrupted!\n");
        throw std::runtime_error("MAX_COLLISIONS_PER_SEGMENT exceeded");
    } else if (error[0] != ERROR_NONE) {
        fprintf(stderr, "ERROR: Undescript error %d\n", error[0]);
        throw std::runtime_error("Indescript error");
    }

    if (printErrors)
        if (error[1] == WARNING_SEGMENT_INTERPENETRATION)
            fprintf(stderr, "WARNING: Interpenetration detection. This can be due to unstable contacts\n");
        else if (error[1] != ERROR_NONE)
            fprintf(stderr, "WARNING: Indescript warning %d\n", error[1]);

    if (error[0] != ERROR_NONE) lastErrorCode = error[0];
    if (error[1] != ERROR_NONE) lastWarningCode = error[1];

    // Reset errors
    if (error[0] != 0 || error[1] != 0) cudaMemsetAsync(d_error, 0, 2 * sizeof(int), stream);
}
__host__ void Sim::iterateCosserat() {
    cosseratItr<<<((meta.numVerts - 1) + (VERTEX_PER_BLOCK - 1)) / (VERTEX_PER_BLOCK - 1), BLOCK_SIZE, 0, stream>>>(d_meta);
    quaternionLambdaItr<<<(meta.numVerts + 127) / 128, 128, 0, stream>>>(d_meta);
}

void Sim::initIterate() { initItr<<<(meta.numVerts + 255) / 256, 256, 0, stream>>>(d_meta); }

void Sim::endIterate() { endItr<<<(meta.numVerts + 255) / 256, 256, 0, stream>>>(d_meta); }

__host__ void Sim::detectCollisions() {
    // Rebuild bvh
    if (meta.useVelocityRadius)
        buildAABBs<true><<<(meta.numVerts + 255) / 256, 256>>>(d_meta, d_error);
    else
        buildAABBs<false><<<(meta.numVerts + 255) / 256, 256>>>(d_meta, d_error);

    //{  // new ver
    //    bvh.compute(meta.d_bounds, meta.numVerts);
    //}

    {  // regacy
        if (lastBVHRebuild >= meta.bvhRebuildPeriod) {
            // printf("compute: \n");
            bvh.compute(meta.d_bounds, meta.numVerts);
            lastBVHRebuild = 0;
        } else {
            // printf("refit: \n");
            bvh.refit();
            lastBVHRebuild += meta.h * meta.detectionPeriod;
        }
    }

    currentBounds = bvh.bounds();

    size_t numCols = bvh.query(meta.d_boundColList, meta.numVerts * MAX_COLLISIONS_PER_SEGMENT);
    bvh.reset();
    // Build collision list
    cudaMemsetAsync(meta.d_numCols, 0, sizeof(int) * meta.numVerts, stream);
    if (meta.useVelocityRadius)
        buildCollisionList<true><<<(numCols + 127) / 128, 128, 0, stream>>>(d_meta, numCols, d_error);
    else
        buildCollisionList<false><<<(numCols + 127) / 128, 128, 0, stream>>>(d_meta, numCols, d_error);
}
__host__ void Sim::recomputeStepLimit() {
    if (meta.useStepSizeLimit) {
        if (meta.useVelocityRadius)
            recomputeStepLimitKernel<true, true><<<(meta.numVerts + 127) / 128, 128, 0, stream>>>(d_meta);
        else
            recomputeStepLimitKernel<true, false><<<(meta.numVerts + 127) / 128, 128, 0, stream>>>(d_meta);
    } else {
        if (meta.useVelocityRadius)
            recomputeStepLimitKernel<false, true><<<(meta.numVerts + 127) / 128, 128, 0, stream>>>(d_meta);
        else
            recomputeStepLimitKernel<false, false><<<(meta.numVerts + 127) / 128, 128, 0, stream>>>(d_meta);
    }
}

__host__ void initStepLimit(MetaData* data, int numVerts) { initStepLimitKernel<<<(numVerts + 127) / 128, 128>>>(data); }

__global__ void initStepLimitKernel(MetaData* data) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= data->numVerts) return;
    data->d_maxStepSize[tid] = 1;
}

void Sim::SagFree(const std::vector<int>& rootVertices, uint32_t iteration) {
    for (uint32_t i = 0; i < iteration; ++i) SingleSagFree(rootVertices);
}

void Sim::SingleSagFree(const std::vector<int>& rootVertices) {
    // #pragma omp parallel for
    for (size_t s = 0; s < rootVertices.size(); ++s) {  // for each strand
        const float eps = 1e-6;
        const vec3 g = meta.gravity;
        int rootv = rootVertices[s];
        int nextv = (s == rootVertices.size() - 1) ? meta.numVerts : rootVertices[s + 1];  // next strand's first vertex
        vec3 f_srs[nextv - rootv]{};  // strength of the bending force exerted by the stretch force

        // ####################1. cancel out the stretch force by modifying rest length ############################
        float totalMassFromTip = 1 / verts[nextv - 1].invMass;

        for (int v = 0; rootv + v < nextv - 1; ++v) {
            const int vId = nextv - 2 - v;  // from tip to root

            const vec3 axis = qs[vId].getD3();  // Get the unit vector that is along the segment.
            const float kStretch = verts[vId].kStretch;
            const float oldRestLen = verts[vId].lRest;

            if (verts[vId].invMass)
                totalMassFromTip += 0.5 / verts[vId].invMass;  // m_(v) / 2
            else {
                totalMassFromTip += 0.5 * (verts[vId].lRest / verts[vId + 1].lRest) / verts[vId + 1].invMass;
            }

            // calc the forces
            double f_s_g = glm::dot(axis, g) * totalMassFromTip;  // the stretch force excerted k-th segment: M_k * g^s_k

            double f_s = -f_s_g;  // k-th stretch force
            double ans1 = 0, ans2 = 0;
            if (f_s == 0) {
                verts[vId].lRest = oldRestLen;
            } else if (1 - 4 * f_s * oldRestLen / kStretch < 0) {
                verts[vId].lRest = 1.5f * oldRestLen;
            } else {
                double fsk = f_s / kStretch;
                double fskinv = kStretch / f_s;
                ans1 = (1 + sqrt(1 - 4 * oldRestLen * fsk)) * 0.5f * fskinv;
                ans2 = (1 - sqrt(1 - 4 * oldRestLen * fsk)) * 0.5f * fskinv;
                verts[vId].lRest = abs(oldRestLen - ans1) >= abs(oldRestLen - ans2) ? ans2 : ans1;  // rounding frequently occurs
            }

            // modify the rest length
            verts[vId].lRest = glm::clamp(verts[vId].lRest, eps, 1.5f * oldRestLen);
            // if (s == 0) {  // for the test
            //     printf("#############ID: %d#######################\n", vId);
            //     printf("kStretch: %.15f,current mass:%.15f ,totalMassFromTip: %.15f\n", kStretch, 1 / verts[vId].invMass, totalMassFromTip);
            //     auto edge = verts[vId + 1].pos - verts[vId].pos;
            //     auto D3 = qs[vId].getD3();
            //     printf("v_k+1 - v_k: (%.10f, %.10f, %.10f)\n", edge.x, edge.y, edge.z);
            //     printf("r:    (%.10f, %.10f, %.10f)\n", axis.x, axis.y, axis.z);
            //     printf("Q.D3: (%.10f, %.10f, %.10f)\n", D3.x, D3.y, D3.z);
            //     auto Mg = g * totalMassFromTip;
            //     printf("Mg: (%.3f, %.15f, %.3f)\n", Mg.x, Mg.y, Mg.z);
            //     printf("f_sr: (%.15f, %.15f, %.15f)\n", f_srs[vId - rootv].x, f_srs[vId - rootv].y, f_srs[vId - rootv].z);
            //     printf("ax^2 + bx + c:(a, b, c) = (%.15f, -1, %.15f)\n", f_s / kStretch, oldRestLen);
            //     printf("D: %.15f\n", 1 - 4 * f_s * oldRestLen / kStretch);
            //     printf("ans1: %.10f, ans2: %.15f\n", ans1, ans2);
            //     printf("ans2:%.15f (double), %.15f (float)\n", ans2, (float)ans2);
            //     printf("Rest length: %.15f (old) -> %.15f (new)\n\n", oldRestLen, verts[vId].lRest);
            // }

            // prepare for the subsequent calculations
            if (rootv + v < nextv - 2) f_srs[vId - rootv - 1] = (float)-f_s * axis;
            totalMassFromTip += 0.5 / verts[vId].invMass;  // m_(v) / 2
        }

        // #################2. cancel out the torque force by modifying rest angle ###############################
        const vec3 up = vec3(0, 1, 0);
        const float nearestAngle = 1e-4;

        float totalMassToTip = 0;
        for (int v = 1; rootv + v < nextv; ++v) {
            if (verts[rootv + v].invMass)
                totalMassToTip += 1 / verts[rootv + v].invMass;
            else
                totalMassToTip += (verts[rootv + v].lRest / verts[rootv + v - 1].lRest) / verts[rootv + v - 1].invMass;
        }

        KTR accumQ = KTR::identity();
        for (int v = 1; rootv + v < nextv - 1; ++v) {  // from root to tip
            const int vId = rootv + v;

            const KTR cQkm1 = qs[vId - 1];     // current q_{k-1}
            const KTR cQk = accumQ * qs[vId];  // current q_k

            if (verts[vId].invMass)
                totalMassToTip -= 0.5 / verts[vId].invMass;  // m_(v) / 2
            else
                totalMassToTip -= 0.5 * (verts[vId].lRest / verts[vId + 1].lRest) / verts[vId + 1].invMass;
            const glm::vec3 axisk = cQk.getD3();                  // the rest axis of q_k
            const glm::vec3 f_b = f_srs[v] + totalMassToTip * g;  // f_i^bt

            // float f_sr_theta = glm::acos(glm::dot(axis, f_sr) / (glm::length(axis) * glm::length(f_sr)));  // for the torque calculation

            if (glm::dot(f_b, axisk) < 1 - nearestAngle) {
                const float kBend = glm::length(qRests[vId]);  // k_i^bt

                glm::vec3 planeNormal = glm::normalize(glm::cross(axisk, f_b));  // n_i
                glm::vec3 torqueDir = glm::cross(planeNormal, axisk);            // t_i

                float torque = glm::dot(torqueDir, f_b) * glm::length(verts[vId + 1].pos - verts[vId].pos);  // T_i
                float T = torque / kBend;                                                                    // T_i/k_i^{bt}

                // regacy
                //// KTR minusF = KTR::fromTo(KTR::identity().getD3(), glm::normalize(-f_b));  //-f_i^bt
                //// the rotations from q^old_{k+1} to -f_b ± nearest angle
                //// KTR maxQ1 = KTR::angleAxis(nearestAngle, planeNormal) * minusF * cQk.inverse();   //\theta_i^{m,\ +\epsilon}
                //// KTR maxQ2 = KTR::angleAxis(-nearestAngle, planeNormal) * minusF * cQk.inverse();  //\theta_i^{m,\ -\epsilon}

                // if (s == 0) {
                //     printf("#############ID: %d#######################\n", vId);
                //     vec3 f_b_norm = f_b / glm::length(f_b);
                //     auto Mg = totalMassToTip * g;
                //     auto fsk1 = f_srs[v];
                //     printf("Mg: (%.15f, %.15f, %.15f)\n", Mg.x, Mg.y, Mg.z);
                //     printf("fsk1: (%.15f, %.15f, %.15f)\n", fsk1.x, fsk1.y, fsk1.z);

                //    printf("f_b: (%.15f, %.15f, %.15f)\n", f_b.x, f_b.y, f_b.z);
                //    printf("f_b_norm: (%.10f, %.10f, %.10f)\n", f_b_norm.x, f_b_norm.y, f_b_norm.z);
                //    KTR f_b_test = minusF.rotate(glm::vec3(0,0,1));
                //    printf("f_b_test: (%.10f, %.10f, %.10f)\n", f_b_test.x, f_b_test.y, f_b_test.z);

                //    printf("r: (%.15f, %.15f, %.15f)\n", axisk.x, axisk.y, axisk.z);
                //    printf("n: (%.15f, %.15f, %.15f)\n", planeNormal.x, planeNormal.y, planeNormal.z);
                //    printf("t: (%.15f, %.15f, %.15f)\n", torqueDir.x, torqueDir.y, torqueDir.z);

                //    printf("maxQ1: (%.10f, %.10f, %.10f, %.10f)\n", maxQ1.v.x, maxQ1.v.y, maxQ1.v.z, maxQ1.v.w);
                //    printf("maxQ2: (%.10f, %.10f, %.10f, %.10f)\n\n", maxQ2.v.x, maxQ2.v.y, maxQ2.v.z, maxQ2.v.w);
                //}

                // calc a closed interval for the optimization.
                dvec3 f_b_norm = f_b / glm::length(f_b);
                double maxrad = glm::acos(glm::dot((glm::dvec3)axisk, -f_b_norm));
                double maxTheta1 = fitAngle(maxrad + nearestAngle), maxTheta2 = fitAngle(maxrad - nearestAngle);

                double maxTheta = abs(maxTheta1) >= abs(maxTheta2) ? maxTheta2 : maxTheta1;  // choosing nearest max angle
                float left = maxTheta >= 0 ? 0 : maxTheta;
                float right = maxTheta >= 0 ? maxTheta : 0;

                double theta1 = -glm::sign(maxTheta) * 100, theta2 = -glm::sign(maxTheta) * 100, resTheta = 0;
                if (abs(T) > 2) {
                    resTheta = maxTheta;
                } else {
                    theta1 = 2 * glm::acos(1 - (0.5 * T * T));
                    theta2 = 2 * glm::acos(-1 + (0.5 * T * T));
                    theta1 = fitAngle(theta1);
                    theta2 = fitAngle(theta2);

                    double cT1 = -glm::sign(maxTheta) * 100, cT2 = -glm::sign(maxTheta) * 100;  // clamped thetas
                    if (abs(theta1) <= myPi2) cT1 = theta1;
                    if (abs(theta2) > myPi2) cT2 = theta2;

                    if (glm::sign(cT1) == glm::sign(maxTheta) && glm::sign(cT2) == glm::sign(maxTheta))
                        resTheta = min(cT1, cT2);
                    else {
                        if (glm::sign(cT1) == glm::sign(maxTheta)) resTheta = cT1;
                        if (glm::sign(cT2) == glm::sign(maxTheta)) resTheta = cT2;
                    }
                }
                resTheta = glm::clamp(resTheta, (double)left, (double)right);

                // modify the rest angle along a plane that contains segment's ends and vertical vec
                KTR qTheta = KTR::angleAxis(resTheta, planeNormal);
                accumQ = accumQ * qTheta;
                qs[vId] = accumQ * qs[vId];
                // if (s == 1000) {  // for the test
                //     printf("#############ID: %d#######################\n", vId);
                //     auto fbt = glm::dot(torqueDir, f_b);
                //     auto l = glm::length(verts[vId + 1].pos - verts[vId].pos);
                //     printf("l: %.15f, fbt:%.15f, T: %.15f\n", l, fbt, torque);
                //     printf("totalMass: %.10f, kBend: %.10f, T:%.15f\n", totalMassToTip, kBend, torque);
                //     printf("r: (%.15f, %.15f, %.15f)\n", axisk.x, axisk.y, axisk.z);
                //     printf("f_b_norm: (%.10f, %.10f, %.10f)\n", f_b_norm.x, f_b_norm.y, f_b_norm.z);
                //     double maxrad = glm::acos(glm::dot((glm::dvec3)axisk, (glm::dvec3)-f_b_norm));
                //     printf("maxtheta from dot: %.15f\n", maxrad * r2d);
                //     printf("maxTheta1:%.4f, maxTheta2:%.4f, maxTheta: %.4f, [%.4f, %.4f]\n", maxTheta1 * r2d, maxTheta2 * r2d, maxTheta * r2d,
                //            left * r2d, right * r2d);
                //     printf("theta1: %.15f, theta2: %.15f, resTheta: %.15f\n\n", theta1 * r2d, theta2 * r2d, resTheta * r2d);
                //     printf("restAngle(before): %.10f ", KTR(qRests[vId - 1]).angle() * r2d);
                // }
                qRests[vId - 1] = glm::normalize((qs[vId - 1].inverse() * qs[vId]).v) * glm::length(qRests[vId - 1]);
                // if (s == 1000) {
                //     printf("restAngle(after): %.10f\n\n", KTR(qRests[vId - 1]).angle() * r2d);
                // }
            }
            if (v == 0)
                totalMassToTip -= (verts[rootv + v].lRest / verts[rootv + v + 1].lRest) / verts[rootv + v + 1].invMass;
            else
                totalMassToTip -= 0.5 / verts[vId].invMass;  // m_(v) / 2
        }
    }
}

}  // namespace YarnBall