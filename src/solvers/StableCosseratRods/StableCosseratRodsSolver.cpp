#include <cuda_runtime.h>
#include <sstream>

#include "StableCosseratRodsSolver.h"
#include "YarnBall/YBSimulator.h"
#include "YarnBall/io/resample.h"
#include "KittenEngine/includes/modules/Dist.h"
#include "KittenEngine/includes/modules/StopWatch.h"
#include "KittenEngine/includes/modules/Rotor.h"
#include "../SDFCollisionSolver/SDFBodyCollisionSolver.h"
#include "../../utilities/Notification.h"
#include "../../utilities/FileUtil.h"
#include "../../utilities/LinearUtil.h"
#include "../../utilities/HairModification.h"

using Kami::LinearUtil::GLMV3f1DArray;

namespace YarnBall {
extern std::shared_ptr<YarnBall::Sim> createFromCurves(vector<vector<glm::vec3>>& curves, vector<bool>& isCurveClosed, int numVerts);
}

[[maybe_unused]]
inline void CkGPUMem() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("%zu KB free of total %zu KB\n", free / 1024, total / 1024);
}

StableCosseratRodsSolver::StableCosseratRodsSolver(wk_ptr<KamiSimulator> _kami) : Solver(_kami) {
    sdfSolver.reset(new SDFBodyCollisionSolver(_kami));
}

void StableCosseratRodsSolver::Construct(string settingJsonPath) {
    SelectCuda();

    if (Kami::FileUtil::IsFileExist(settingJsonPath)) {
        jsonParams = LoadSettingJsonContent(settingJsonPath);
    }

    if (jsonParams.SubdivCoeff > 1) Kami::HairMods::DivideHair(GetHair().lock(), jsonParams.SubdivCoeff);

    if (Kami::FileUtil::IsFileExist(jsonParams.YBJsonPath)) {
        sim = YarnBall::buildFromJSON(jsonParams.YBJsonPath);
    } else {
        sim = BuildYarnBallHair(jsonParams);
    }

    // printf("Total verts: %d\n", sim->meta.numVerts);
    sim->printErrors = false;
    sim->renderShaded = true;

    for (int i = 0; i < sim->meta.numVerts; i++) {
        auto pos = sim->verts[i].pos;
        initialBounds.absorb(pos);
    }
    if (jsonParams.EnableSDF) {
        sdfSolver->SubstituteHair(GetHair());  // propagate SubstituteHair() for cantilTest
        sdfSolver->SubstituteBody(GetBody());  // propagate SubstituteBodyy() for cantilTest
        sdfSolver->Construct();
        // sdfSolver->EnableStrainLimiting(true);
    }
}

int StableCosseratRodsSolver::SelectCuda() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device = deviceCount - 1;
    cudaError_t cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        string err = to_string(cudaStatus);
        Kami::Notification::Warn(__func__, "cudaSetDevice failed! Error: " + err + "\n");
    }
    return device;
}

StableCosseratRodsSolver::JsonParams StableCosseratRodsSolver::LoadSettingJsonContent(string jsonPath) {
    JsonParser jp(jsonPath);
    jDump = jp.dump();
    JsonParams jParams{};

    // options
    if (jp.CheckKey("YBJsonPath")) jParams.YBJsonPath = jp.GetString("YBJsonPath");
    if (jp.CheckKey("EnableResample")) jParams.EnableResample = jp.GetBool("EnableResample");
    if (jp.CheckKey("EnableSDF")) jParams.EnableSDF = jp.GetBool("EnableSDF");
    if (jp.CheckKey("EnableHairHairColl")) jParams.EnableHairHairColl = jp.GetBool("EnableHairHairColl");
    if (jp.CheckKey("UseYoungsModule")) jParams.UseYoungsModule = jp.GetBool("UseYoungsModule");

    // sim params
    if (jp.CheckKey("NumItr")) jParams.NumItr = jp.GetInteger("NumItr");
    if (jp.CheckKey("SagFreeItr")) jParams.SagFreeItr = jp.GetInteger("SagFreeItr");
    if (jp.CheckKey("SubdivCoeff")) jParams.SubdivCoeff = jp.GetInteger("SubdivCoeff");
    if (jp.CheckKey("BendK")) jParams.BendK = jp.GetFloat("BendK");
    if (jp.CheckKey("StretchK")) jParams.StretchK = jp.GetFloat("StretchK");
    if (jp.CheckKey("CurveRadius")) jParams.CurveRadius = jp.GetFloat("CurveRadius");
    if (jp.CheckKey("Density")) jParams.Density = jp.GetFloat("Density");
    if (jp.CheckKey("YoungsModule")) jParams.YoungsModule = jp.GetFloat("YoungsModule");
    if (jp.CheckKey("StiffnessDeviationCoeff")) jParams.StiffnessDeviationCoeff = jp.GetFloat("StiffnessDeviationCoeff");
    if (jp.CheckKey("ResampleLen")) jParams.ResampleLen = jp.GetFloat("ResampleLen");
    if (jp.CheckKey("VelocityDecay")) jParams.VelocityDecay = jp.GetFloat("VelocityDecay");
    if (jp.CheckKey("ExternalForceMultiplier")) jParams.ExternalForceMultiplier = jp.GetFloat("ExternalForceMultiplier");

    // wind params
    if (jp.CheckKey("WindPeak")) jParams.WindPeak = jp.GetFloat("WindPeak");
    if (jp.CheckKey("WindYFreq")) jParams.WindYFreq = jp.GetFloat("WindYFreq");
    if (jp.CheckKey("WindZFreq")) jParams.WindZFreq = jp.GetFloat("WindZFreq");
    if (jp.CheckKey("WindTimeFreq")) jParams.WindTimeFreq = jp.GetFloat("WindTimeFreq");
    if (jp.CheckKey("WindSharpness")) jParams.WindSharpness = jp.GetFloat("WindSharpness");

    return jParams;
}

sh_ptr<YarnBall::Sim> StableCosseratRodsSolver::BuildYarnBallHair(JsonParams jp) {
    auto hair = GetHair();
    auto hairSize = hair.lock()->GetHairParams().hairSize;
    auto& strandVertCount = hair.lock()->GetStrandVertCountRef();
    auto& strands = hair.lock()->GetCurrentVerticesRef();

    glm::mat4 transform(1);
    transform[0][0] = transform[1][1] = transform[2][2] = 0.01f;

    //// Build curve from class Hair
    vector<vector<glm::vec3>> curves;
    vector<bool> isCurveClosed;

    int numVerts = 0;

    for (uint32_t i = 0; i < hairSize.x; i++) {
        vector<glm::vec3> curve;

        for (uint32_t j = 0; j < strandVertCount[i]; j++) {
            curve.push_back(strands->GetEntryVal(i, j));
        }

        if (curve.size() < 4) {
            continue;
        }

        if (jp.EnableResample) {
            curve = Resample::resampleCMR(curve, 1, curve.size() - 2, jp.ResampleLen);
            auto originalLen = strandVertCount[i];
            auto newLen = curve.size();

            if (hairSize.y < newLen) {
                vector<glm::vec3> newCurve(hairSize.y);
                float stride = (float)(newLen - 2) / (float)(hairSize.y - 2);

                // resize curve
                newCurve[0] = curve[0];
                for (uint32_t v = 1; v < hairSize.y - 1; ++v) {
                    newCurve[v] = curve[(uint32_t)(stride * v)];
                }
                newCurve[hairSize.y - 1] = curve[curve.size() - 1];

                curve.resize(newCurve.size());
                curve = newCurve;
            }

            strandVertCount[i] = curve.size();
            *(hair.lock()->GetVerticesCountRef()) += (curve.size() - originalLen);

            for (uint32_t v = 0; v < curve.size(); ++v) {
                strands->SetEntryToArray(curve[v], i, v);
            }
        }

        curves.push_back(curve);
        isCurveClosed.push_back(false);

        // for write back hair
        rootVertices.push_back(numVerts);
        strandRefs.push_back(i);
        numVerts += strandVertCount[i];
    }

    sim = YarnBall::createFromCurves(curves, isCurveClosed, numVerts);

    float r = jp.CurveRadius;
    constexpr float ratio = 0.05f;
    sim->meta.radius = ratio * r;
    sim->meta.barrierThickness = 2 * (1 - ratio) * r;

    sim->meta.numItr = jp.NumItr;
    sim->meta.velocityDecay = jp.VelocityDecay;
    sim->meta.useStepSizeLimit = jp.EnableHairHairColl;
    sim->meta.useVelocityRadius = true;
    sim->meta.enableHairHairColl = jp.EnableHairHairColl;
    sim->meta.ExternalForceMultiplier = jp.ExternalForceMultiplier;
    sim->meta.windPeak = jp.WindPeak;
    sim->meta.WindYFreq = jp.WindYFreq;
    sim->meta.WindZFreq = jp.WindZFreq;
    sim->meta.windTimeFreq = jp.WindTimeFreq;
    sim->meta.windSharpness = jp.WindSharpness;

    sim->configure(jp.Density);

    if (jp.UseYoungsModule) {
        sim->setYoung(jp.YoungsModule, jp.StiffnessDeviationCoeff);
    } else {
        sim->setKBend(jp.BendK);
        sim->setKStretch(jp.StretchK);
    }

    for (uint32_t i = 0; i < rootVertices.size(); ++i) {  // set root
        sim->verts[rootVertices[i]].invMass = 0;
    }

    if (jp.SagFreeItr) sim->SagFree(rootVertices, jp.SagFreeItr);

    sim->upload();

    return sim;
}

void StableCosseratRodsSolver::ModificateRoot() {}

void StableCosseratRodsSolver::SolveNextStep(float dt) {
    Kit::StopWatch timer;

    float measuredTime = timer.time();

    sim->advance(dt);

    float ss = dt / measuredTime;
    simSpeedDist->accu(ss);

    // Actually, it is preferable to resolve collisions in VBD's style.
    DownloadHair();
    if (jsonParams.EnableSDF) {
        sdfSolver->UploadHair();
        sdfSolver->SolveNextStep(dt);
        sdfSolver->DownloadHair();
        UploadHair();
    }
}

StableCosseratRodsSolver::~StableCosseratRodsSolver() {}

void StableCosseratRodsSolver::UploadHair() {
    auto hair = GetHair();
    auto& positions = hair.lock()->GetCurrentVerticesRef();
    [[maybe_unused]]
    auto& velocities = hair.lock()->GetCurrentVelocitiesRef();

#pragma omp parallel for
    for (uint32_t i = 0; i < strandRefs.size(); ++i) {
        int u = strandRefs[i];
        int rootv = rootVertices[i];
        int nextv = (i == strandRefs.size() - 1) ? sim->meta.numVerts : rootVertices[i + 1];

        for (int v = 0; rootv + v < nextv; ++v) {
            sim->verts[rootv + v].pos = positions->GetEntryVal(u, v);
            sim->vels[rootv + v] = velocities->GetEntryVal(u, v);
        }
    }
    sim->uploadPosAndVel();
}

void StableCosseratRodsSolver::DownloadHair() {
    sim->downloadPosAndVel();
    auto hair = GetHair();
    auto& positions = hair.lock()->GetCurrentVerticesRef();
    auto& velocities = hair.lock()->GetCurrentVelocitiesRef();
#pragma omp parallel for
    for (uint32_t i = 0; i < strandRefs.size(); ++i) {
        int u = strandRefs[i];
        int rootv = rootVertices[i];

        int nextv = (i == strandRefs.size() - 1) ? sim->meta.numVerts : rootVertices[i + 1];

        for (int v = 0; (rootv + v) < nextv; ++v) {
            auto vert = sim->verts[rootv + v];
            auto vel = sim->vels[rootv + v];
            positions->SetEntryToArray(vert.pos, u, v);
            velocities->SetEntryToArray(vel, u, v);
        }
    }
}

string StableCosseratRodsSolver::GetInfoString() {
    std::stringstream sstr;

    sstr << "Solver: Stable Cosserat Rods" << endl;
    sstr << "SubdivCoeff: " << jsonParams.SubdivCoeff << endl;

    if (jsonParams.UseYoungsModule) {
        sstr << "Use Youngs: " << "Yes" << endl;
        sstr << "Youngs: " << jsonParams.YoungsModule << endl;
        sstr << "Deviation: " << jsonParams.StiffnessDeviationCoeff << endl;
    } else {
        sstr << "Use Youngs: " << "No" << endl;
        sstr << "StretchK: " << jsonParams.StretchK << endl;
        sstr << "BendK: " << jsonParams.BendK << endl;
    }
    sstr << "Sag-free itr: " << jsonParams.SagFreeItr << endl;

    sstr << "Json:" << endl;

    sstr << jDump;

    return sstr.str();
}
