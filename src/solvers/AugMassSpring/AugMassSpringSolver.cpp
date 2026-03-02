#include "AugMassSpringSolver.h"
#include "HairGen.h"
#include "solver.h"
#include "SimpleLoader.h"
#include "../../../extern/json.hpp"

#include "../../utilities/Notification.h"
#include "../../utilities/LinearUtil.h"
#include "../../utilities/JsonParser.h"
#include "../../utilities/FileUtil.h"

// DigitalSalon: A Fast Simulator for Dense Hair via Augmented Mass-Spring Model
//( https://repository.kaust.edu.sa/items/b4a5000e-58c8-4b94-aa88-0e2638884b4c )

class Body;
class HairGenerator;

using namespace Kami;
using std::shared_ptr;

void AugMassSpringSolver::Construct(string settingJsonPath) {
    auto headmeshInfo = GetBody().lock()->GetCurrentHeadOrBoxPNMeshPtr()->GetPNMeshParam();
    auto headSdfDim = headmeshInfo.SDFDim;
    // Set hairinfo
    hairInfo.FilePath = GetHair().lock()->GetHairParams().srcFilePath;
    hairInfo.Move = make_float4(0.f, 0.f, 0.f, 1.f);
    hairInfo.ID = 0;

    // Set headinfo
    headInfo.FilePath = headmeshInfo.meshPath;
    headInfo.Move = make_float4(0.f, 0.f, 0.f, 1.f);
    headInfo.SDFDim = make_int3(headSdfDim.x, headSdfDim.y, headSdfDim.z);
    headInfo.Animate = false;
    headInfo.NumFrames = 1;

    amsSolver.reset(new AMSSolver(GetHair(), GetBody()));

    shared_ptr<SimpleObject> obj = make_shared<SimpleObject>(GetBody(), GetKamiParams().OutputNamePrefix);
    meshes.push_back(obj);

    hairGen = make_shared<HairGenerator>(meshes);
    // Sdf parity (to avoid buggs if mesh was inverted)
    amsSolver->GetParamsSimRef().Parity = headInfo.Move.w < 0 ? -1.f : 1.f;
    amsSolver->HeadInformation = headInfo;

    if (Kami::FileUtil::IsFileExist(settingJsonPath)) LoadSettingJsonContent(settingJsonPath);

    hairGen->LoadHair(GetHair(), GetBody().lock()->GetCurrentHeadOrBoxPNMeshPtr()->GetPNMeshParam().masterTrans);

    GenerateHairAndHeadGrid();
    hairGen->GenerateParticles();
    amsSolver->InitBuffers(meshes, hairGen);
    amsSolver->InitParticles();
    amsSolver->ComputeSDFCPU(GetBody());
    // amsSolver->ComputeBary();

    // Clears memory
    meshes.clear();
    hairGen->Clear();
    isConstructed = true;
}

void AugMassSpringSolver::UseSDFOrNot(bool yn) { amsSolver->UseSdfOrNot(yn); }

void AugMassSpringSolver::LoadSettingJsonContent(string jsonPath) {
    JsonParser jp(jsonPath);
    JsonParams jParams;

    if (jp.CheckKey("WindPeak")) jParams.WindPeak = jp.GetFloat("WindPeak");
    if (jp.CheckKey("SDFThreshold")) jParams.SDFThreshold = jp.GetFloat("SDFThreshold");

    AMSSolver::ModParams amsPara;

    amsPara.windPeak = jParams.WindPeak;
    amsPara.SDFThreshold = jParams.SDFThreshold;
    amsSolver->ModifyParams(amsPara);
}

void AugMassSpringSolver::GenerateHairAndHeadGrid() {
    // ###########################################################################
    // get grids' dimensions
    // ###########################################################################

    // hairGrid dim
    int nx = amsSolver->GetParamsSimRef().HairDim.x;
    int ny = amsSolver->GetParamsSimRef().HairDim.y;
    int nz = amsSolver->GetParamsSimRef().HairDim.z;

    // mesh grid dim
    int nxS = amsSolver->GetParamsSimRef().SDFDim.x;
    int nyS = amsSolver->GetParamsSimRef().SDFDim.y;
    int nzS = amsSolver->GetParamsSimRef().SDFDim.z;

    // ###########################################################################
    // The grids will be expand (1 + gridRescaleFactor) times
    // ###########################################################################
    float gridRescaleFactor = amsSolver->GetParamsSimRef().LengthIncreaseGrid;

    // ###########################################################################
    // grid expansion
    // ###########################################################################

    // hair
    Kami::AABB HairAABB = GetHair().lock()->GetAABB();
    float3 hairMin = make_float3(HairAABB.Xmin, HairAABB.Ymin, HairAABB.Zmin);
    float3 hairMax = make_float3(HairAABB.Xmax, HairAABB.Ymax, HairAABB.Zmax);
    float3 hairDistances = hairMax - hairMin;
    float3 hairDimIncrease = 0.5f * gridRescaleFactor * hairDistances;

    vector<float3> hairNewMinMax = {hairMin - hairDimIncrease, hairMax + hairDimIncrease};
    float3 hairNewDist = hairNewMinMax[1] - hairNewMinMax[0];

    // head
    vector<float3> SDFMinMax = meshes[0]->GetAABB();
    float3 SDFDistances = SDFMinMax[1] - SDFMinMax[0];
    // float3 SDFdimIncrease = 0.5f * gridRescaleFactor * SDFDistances;

    // vector<float3> SDFnewMinMax = {SDFMinMax[0] - SDFdimIncrease, SDFMinMax[1] + SDFdimIncrease};
    // float3 SDFnewDist = (SDFnewMinMax[1] - SDFnewMinMax[0]);

    // ###########################################################################
    // grid units preparation
    // ###########################################################################
    // hair
    float dx = hairNewDist.x / (1.f * nx);  //!
    float dy = hairNewDist.y / (1.f * ny);  //!
    float dz = hairNewDist.z / (1.f * nz);  //!

    // head
    float dxS = SDFDistances.x / (1.f * nxS);
    float dyS = SDFDistances.y / (1.f * nyS);
    float dzS = SDFDistances.z / (1.f * nzS);
    // ###########################################################################
    // Set parameters before bone transform
    // ###########################################################################

    // hair
    amsSolver->GetParamsSimRef().HairCenter0 = 0.5 * (hairNewMinMax[1] + hairNewMinMax[0]);  //!
    amsSolver->GetParamsSimRef().HairAxis0[0] = make_float3(1.f, 0.f, 0.f);
    amsSolver->GetParamsSimRef().HairAxis0[1] = make_float3(0.f, 1.f, 0.f);
    amsSolver->GetParamsSimRef().HairAxis0[2] = make_float3(0.f, 0.f, 1.f);
    amsSolver->GetParamsSimRef().HairMin0 = hairNewMinMax[0];  //!

    // head
    amsSolver->GetParamsSimRef().HeadCenter0 = 0.5 * (SDFMinMax[1] + SDFMinMax[0]);
    amsSolver->GetParamsSimRef().HeadAxis0[0] = make_float3(1.f, 0.f, 0.f);
    amsSolver->GetParamsSimRef().HeadAxis0[1] = make_float3(0.f, 1.f, 0.f);
    amsSolver->GetParamsSimRef().HeadAxis0[2] = make_float3(0.f, 0.f, 1.f);
    amsSolver->GetParamsSimRef().HeadMin0 = SDFMinMax[0];

    // ###########################################################################
    // After bone transform (identity at t0)
    // ###########################################################################

    // hair
    amsSolver->GetParamsSimRef().HairCenter = 0.5 * (hairNewMinMax[1] + hairNewMinMax[0]);
    amsSolver->GetParamsSimRef().HairAxis[0] = make_float3(1.f, 0.f, 0.f);
    amsSolver->GetParamsSimRef().HairAxis[1] = make_float3(0.f, 1.f, 0.f);
    amsSolver->GetParamsSimRef().HairAxis[2] = make_float3(0.f, 0.f, 1.f);
    amsSolver->GetParamsSimRef().HairMin = hairNewMinMax[0];

    // head
    amsSolver->GetParamsSimRef().HeadCenter = 0.5 * (SDFMinMax[1] + SDFMinMax[0]);
    amsSolver->GetParamsSimRef().HeadAxis[0] = make_float3(1.f, 0.f, 0.f);
    amsSolver->GetParamsSimRef().HeadAxis[1] = make_float3(0.f, 1.f, 0.f);
    amsSolver->GetParamsSimRef().HeadAxis[2] = make_float3(0.f, 0.f, 1.f);
    amsSolver->GetParamsSimRef().HeadMin = SDFMinMax[0];

    // ###########################################################################
    // Static params
    // ###########################################################################

    // hair
    amsSolver->GetParamsSimRef().MaxWeight = 1.5 * max(dx, max(dy, dz)) * (amsSolver->GetParamsSimRef().NumGridNeighbors + 1.f);
    amsSolver->GetParamsSimRef().HairInvSqDs = make_float3(1.f / (dx * dx), 1.f / (dy * dy), 1.f / (dz * dz));
    amsSolver->GetParamsSimRef().HairInvDs = make_float3(1.f / dx, 1.f / dy, 1.f / dz);
    amsSolver->GetParamsSimRef().HairDs = make_float3(dx, dy, dz);

    // head
    amsSolver->GetParamsSimRef().SdfInvSqDs = make_float3(1.f / (dxS * dxS), 1.f / (dyS * dyS), 1.f / (dzS * dzS));
    amsSolver->GetParamsSimRef().SDFInvDs = make_float3(1.f / dxS, 1.f / dyS, 1.f / dzS);
    amsSolver->GetParamsSimRef().SdfDs = make_float3(dxS, dyS, dzS);

    // cout << "Sanity Check" << endl;
    // printf("Sdf Dim (%i,%i,%i)\n", nxS, nyS, nzS);
    //  printf("Extents (%f,%f, %f)  (%f, %f,%f)\n", newMinMax[0].x, newMinMax[0].y, newMinMax[0].z,
    //      newMinMax[1].x, newMinMax[1].y, newMinMax[1].z);
    // printf("Ds (%f,%f,%f)\n", dxS, dyS, dzS);
}

void AugMassSpringSolver::SolveNextStep(float dt) {
    if (!isConstructed) {
        Kami::Notification::Warn(__func__, "Please call \033[31m Construct() \033[0m before call this.");
    } else {
        amsSolver->GetParamsSimRef().Dt = dt;
        amsSolver->UpdateSimulation(GetHair(), GetBody());
    }
}

void AugMassSpringSolver::UploadHair() { amsSolver->UploadHair(GetHair(), GetBody()); }
void AugMassSpringSolver::DownloadHair() { amsSolver->DownloadHair(GetHair(), GetBody()); }