#include "SDFBodyCollisionSolver.h"

#include "../../utilities/JsonParser.h"
#include "../../utilities/FileUtil.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "../../../extern/cuda_helpers/helper_cuda.h"
#include "../../../extern/glm/glm/glm.hpp"
#include "../../../extern/glm/glm/gtx/matrix_decompose.hpp"

void SDFBodyCollisionSolver::Construct(string settingJsonPath) {
    JsonParams jParams{};
    if (Kami::FileUtil::IsFileExist(settingJsonPath)) {
        jParams = LoadSetting(settingJsonPath);
    }
    InitCudaDevice();
    InitParams(jParams);
    InitBuffers();

    isConstructed = true;
}

void SDFBodyCollisionSolver::InitCudaDevice() {
    // init cuda device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device = deviceCount - 1;
    checkCudaErrors(cudaSetDevice(device));
}

SDFBodyCollisionSolver::JsonParams SDFBodyCollisionSolver::LoadSetting(string settingJsonPath) {
    JsonParser jp(settingJsonPath);
    JsonParams jParams{};

    if (jp.CheckKey("SDFThreshold")) jParams.SDFThreshold = jp.GetFloat("SDFThreshold");
    if (jp.CheckKey("StrainError")) jParams.StrainError = jp.GetFloat("StrainError");
    if (jp.CheckKey("UseStrainLimiting")) jParams.UseStrainLimiting = jp.GetBool("UseStrainLimiting");

    return jParams;
}

inline Eigen::Vector3f Float2eigen(const float3& u) { return Eigen::Vector3f(u.x, u.y, u.z); }
inline glm::vec3 CFloat3ToGLMV3(const float3& u) { return glm::vec3(u.x, u.y, u.z); }
inline float3 GLMV3ToCFloat3(const glm::vec3 v) { return make_float3(v.x, v.y, v.z); }

void SDFBodyCollisionSolver::InitParams(SDFBodyCollisionSolver::JsonParams jParams) {
    auto hSDFsize = GetBody().lock()->GetCurrentHeadOrBoxPNMeshPtr()->GetPNMeshParam().SDFDim;
    auto hairInfo = GetHair().lock()->GetHairParams();

    simParams.Dt = GetKamiInternalStepSize();
    simParams.SDFDim = make_int3(hSDFsize.x, hSDFsize.y, hSDFsize.z);

    simParams.NumStrands = hairInfo.hairSize.x;
    simParams.MaxStrandLength = hairInfo.hairSize.y;
    simParams.NumParticles = GetHair().lock()->GetVerticesCount();
    simParams.NumSdfCells = simParams.SDFDim.x * simParams.SDFDim.y * simParams.SDFDim.z;

    simParams.ThreshSdf = jParams.SDFThreshold;
    simParams.StrainError = jParams.StrainError;
    simParams.UseStrainLimiting = jParams.UseStrainLimiting;

    // SDF grid's parameters
    auto headmesh = GetBody().lock()->GetCurrentHeadOrBoxPNMeshPtr();
    auto headAABB = headmesh->GetPNMeshParam().meshAABB;

    int nxS = simParams.SDFDim.x;
    int nyS = simParams.SDFDim.y;
    int nzS = simParams.SDFDim.z;

    vector<float3> SDFMinMax = {make_float3(headAABB.Xmin, headAABB.Ymin, headAABB.Zmin),  // 0: min, 1: max
                                make_float3(headAABB.Xmax, headAABB.Ymax, headAABB.Zmax)};
    float3 SDFDistances = SDFMinMax[1] - SDFMinMax[0];

    float dxS = SDFDistances.x / (1.f * nxS);
    float dyS = SDFDistances.y / (1.f * nyS);
    float dzS = SDFDistances.z / (1.f * nzS);

    simParams.HeadMin = SDFMinMax[0];
    simParams.HeadMax = SDFMinMax[1];

    simParams.SdfInvSqDs = make_float3(1.f / (dxS * dxS), 1.f / (dyS * dyS), 1.f / (dzS * dzS));
    simParams.SDFInvDs = make_float3(1.f / dxS, 1.f / dyS, 1.f / dzS);
    simParams.SdfDs = make_float3(dxS, dyS, dzS);

    // Compute block sizes for CUDA
    BlockThreadHair.x = DivUp(simParams.NumParticles, BlockThreadHair.y);
    BlockThreadSdf.x = DivUp(simParams.NumSdfCells, BlockThreadSdf.y);
    BlockThreadRoot.x = DivUp(simParams.NumStrands, BlockThreadRoot.y);

    KamiCopySDFSimParamsToDevice(&simParams);
}

void SDFBodyCollisionSolver::InitAdditionalHairData() {
    auto hair = GetHair();
    auto& hairVerts = hair.lock()->GetCurrentVerticesRef();
    auto& strandVertCount = hair.lock()->GetStrandVertCountRef();

    vector<float> restLengthsVec;
    vector<int> rootIdxVec(simParams.NumStrands);

    // prepare rootIdx
    int counter = 0;
    for (size_t i = 0; i < simParams.NumStrands; i++) {
        rootIdxVec[i] = counter;
        counter += strandVertCount[i];
    }

    // restLengthsVec's structure is left unchanged from the original one
    for (size_t i = 0; i < simParams.NumStrands; ++i) {
        for (size_t j = 0; j < strandVertCount[i]; ++j) {
            auto currentVert = hairVerts->GetEntryVal(i, j);
            auto currentIdx = rootIdxVec[i] + j;

            // edge
            if (j + 1 < strandVertCount[i]) {
                auto adjVert = hairVerts->GetEntryVal(i, j + 1);

                float l = length(adjVert - currentVert);
                restLengthsVec.push_back(l);
                particlesBuffer->HostPtr()[currentIdx].EdgeRestIdx.y = restLengthsVec.size() - 1;
                particlesBuffer->HostPtr()[currentIdx + 1].EdgeRestIdx.x = restLengthsVec.size() - 1;
            }

            // bending
            if (j + 2 < strandVertCount[i]) {
                auto adjVert = hairVerts->GetEntryVal(i, j + 2);

                float l = length(adjVert - currentVert);
                restLengthsVec.push_back(l);
                particlesBuffer->HostPtr()[currentIdx].BendRestIdx.y = restLengthsVec.size() - 1;
                particlesBuffer->HostPtr()[currentIdx + 2].BendRestIdx.x = restLengthsVec.size() - 1;
            }

            // torsion
            if (j + 3 < strandVertCount[i]) {
                auto adjVert = hairVerts->GetEntryVal(i, j + 3);

                float l = length(adjVert - currentVert);
                restLengthsVec.push_back(l);
                particlesBuffer->HostPtr()[currentIdx].TorsRestIdx.y = restLengthsVec.size() - 1;
                particlesBuffer->HostPtr()[currentIdx + 3].TorsRestIdx.x = restLengthsVec.size() - 1;
            }
        }
    }
    // copy rengths and indices to device
    restLengthsBuffer = make_shared<CuBuffer<float>>(restLengthsVec.size());
    rootIdxBuffer = make_shared<CuBuffer<int>>(rootIdxVec.size(), true);

    deviceBuffers.restLenghts = restLengthsBuffer->DevPtr();
    deviceBuffers.rootIdx = rootIdxBuffer->DevPtr();

    restLengthsBuffer->CopyHostToDevice(restLengthsVec.data());
    rootIdxBuffer->CopyHostToDevice(rootIdxVec.data());

    // Respect the Original intention
    rootIdxBuffer->CopyDeviceToHost();
}

void SDFBodyCollisionSolver::InitBuffers() {
    // hair
    particlesBuffer = make_shared<CuBuffer<ParticleForSDFColl>>(simParams.NumParticles, true);
    deviceBuffers.Particles = particlesBuffer->DevPtr();

    InitAdditionalHairData();

    //// Head
    SDFBuffer = make_shared<CuBuffer<float>>(simParams.NumSdfCells, true);
    nablaSDFBuffer = make_shared<CuBuffer<float3>>(simParams.NumSdfCells, true);
    headVelXBuffer = make_shared<CuBuffer<float>>(simParams.NumSdfCells);
    headVelYBuffer = make_shared<CuBuffer<float>>(simParams.NumSdfCells);
    headVelZBuffer = make_shared<CuBuffer<float>>(simParams.NumSdfCells);

    deviceBuffers.SDF = SDFBuffer->DevPtr();
    deviceBuffers.NablaSDF = nablaSDFBuffer->DevPtr();
    deviceBuffers.HeadVelX = headVelXBuffer->DevPtr();
    deviceBuffers.HeadVelY = headVelYBuffer->DevPtr();
    deviceBuffers.HeadVelZ = headVelZBuffer->DevPtr();

    KamiLaunchKernelInitHeadSDFVel(BlockThreadSdf, deviceBuffers);
}

void SDFBodyCollisionSolver::SolveNextStep(float dt) {
    SendSimParamsToCuda(dt);
    SendLatestSDFToCuda();
    KamiLaunchKernelSDFHeadCollision(BlockThreadHair, deviceBuffers);  // solve
    if (simParams.UseStrainLimiting) KamiLaunchKernelStrainLimitingAfterSDFCollision(BlockThreadRoot, deviceBuffers);
}

void SDFBodyCollisionSolver::EnableStrainLimiting(bool yn) { simParams.UseStrainLimiting = yn; }

void SDFBodyCollisionSolver::UploadHair() {
    auto hair = GetHair();
    auto& hVerts = hair.lock()->GetCurrentVerticesRef();
    auto& hVels = hair.lock()->GetCurrentVelocitiesRef();
    auto& numStrands = hair.lock()->GetStrandVertCountRef();

#pragma omp parallel for
    for (size_t i = 0; i < simParams.NumStrands; i++) {
        auto rootIdx = rootIdxBuffer->HostPtr()[i];
#pragma omp parallel for
        for (size_t j = 0; j < numStrands[i]; j++) {
            ParticleForSDFColl& p = particlesBuffer->HostPtr()[rootIdx + j];
            p.Position = GLMV3ToCFloat3(hVerts->GetEntryVal(i, j));
            p.Velocity = GLMV3ToCFloat3(hVels->GetEntryVal(i, j));
            p.StrandLength = numStrands[i];
            p.LocalIdx = j;
            p.GlobalIdx = rootIdx + j;
        }
    }
    particlesBuffer->CopyHostToDevice();
}

void SDFBodyCollisionSolver::DownloadHair() {
    auto hair = GetHair();
    auto& hVerts = hair.lock()->GetCurrentVerticesRef();
    auto& hVels = hair.lock()->GetCurrentVelocitiesRef();
    auto& numStrands = hair.lock()->GetStrandVertCountRef();

    particlesBuffer->CopyDeviceToHost();

#pragma omp parallel for
    for (size_t i = 0; i < simParams.NumStrands; i++) {
#pragma omp parallel for
        for (size_t j = 0; j < numStrands[i]; j++) {
            int rootIdx = rootIdxBuffer->HostPtr()[i];
            hVerts->SetEntryToArray(CFloat3ToGLMV3(particlesBuffer->HostPtr()[rootIdx + j].Position), i, j);
            hVels->SetEntryToArray(CFloat3ToGLMV3(particlesBuffer->HostPtr()[rootIdx + j].Velocity), i, j);
        }
    }
}

void SDFBodyCollisionSolver::SendSimParamsToCuda(float dt) {
    auto headmeshInfo = GetBody().lock()->GetCurrentHeadOrBoxPNMeshPtr()->GetPNMeshParam();

    simParams.Dt = dt;
    simParams.InvDt = 1.f / dt;

    // Update Head axises
    glm::vec3 scale;
    glm::quat rotation;
    glm::vec3 translate = glm::vec3(0, 0, 0);
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(headmeshInfo.masterTrans, scale, rotation, translate, skew, perspective);

    simParams.rotation = make_float4(rotation.x, rotation.y, rotation.z, rotation.w);
    simParams.translate = make_float3(translate.x, translate.y, translate.z);
    KamiCopySDFSimParamsToDevice(&simParams);
}

void SDFBodyCollisionSolver::RecieveHairFromCuda() {}

void SDFBodyCollisionSolver::SendLatestSDFToCuda() {
    auto pnMesh = GetBody().lock()->GetCurrentHeadOrBoxPNMeshPtr();
    if (pnMesh->GetSDFCreatedTime() != *sdfCreatedTime) {
        vector<float> ArraySDF;
        vector<float3> ArrayNabla;
        vector<float3> pos;

        pnMesh->StoreSDFQueries(pos, ArraySDF, ArrayNabla, sdfCreatedTime);

#pragma omp parallel for
        for (uint32_t i = 0; i < simParams.NumSdfCells; i++) {
            SDFBuffer->HostPtr()[i] = ArraySDF[i];
            nablaSDFBuffer->HostPtr()[i] = ArrayNabla[i];
        }

        // Copy to GPU
        SDFBuffer->CopyHostToDevice();
        nablaSDFBuffer->CopyHostToDevice();
    }
}

int SDFBodyCollisionSolver::DivUp(const int& a, const int& b) { return (a % b != 0) ? (a / b + 1) : (a / b); }