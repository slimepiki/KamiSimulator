#include "HairModification.h"
#include "Notification.h"
#include <algorithm>

inline glm::vec3 interpolate(glm::vec3 x, glm::vec3 y, int den, int num) {
    float fden = den;
    float fnum = num;

    return ((fden - fnum) * x + fnum * y) / fden;
}

void Kami::HairMods::DivideHair(sh_ptr<Hair> hair, size_t mul) {
    if (mul > 20) {
        Kami::Notification::Warn(__func__, "mul should be <= 20.");
    }
    if (mul > 3) {
        Kami::Notification::Caution(__func__, "Hair size will be huge.");
    }

    auto& baseInitialVerts = hair->GetInitialVerticesRef();
    auto& baseCurrentVerts = hair->GetCurrentVerticesRef();
    auto& baseStrandVertsCount = hair->GetStrandVertCountRef();
    auto& baseCurrentVels = hair->GetCurrentVelocitiesRef();
    auto baseSize = hair->GetHairParams().hairSize;

    sh_ptr<Hair> resHair = make_shared<Hair>();
    Hair::HairParams resParams;
    resParams.hairSize = glm::uvec2(baseSize.x, (baseSize.y - 1) * mul + 1);
    resParams.srcFilePath = hair->GetHairParams().srcFilePath;

    auto& resInitialVerts = resHair->GetInitialVerticesRef();
    auto& resCurrentVerts = resHair->GetCurrentVerticesRef();
    auto& resStrandVertCount = resHair->GetStrandVertCountRef();
    auto& resCurrentVels = resHair->GetCurrentVelocitiesRef();
    auto& resVertCount = resHair->GetVerticesCountRef();

    resInitialVerts = make_unique<EigenVecX3f2DArray>(resParams.hairSize);
    resCurrentVerts = make_unique<EigenVecX3f2DArray>(resParams.hairSize);
    resCurrentVels = make_unique<EigenVecX3f2DArray>(resParams.hairSize);
    resStrandVertCount = make_unique<uint32_t[]>(resParams.hairSize.x);
    resHair->GetAABBRef() = make_unique<AABB>();
    resHair->GetHairParamsRef() = resParams;

    *(resHair->GetAABBRef()) = hair->GetAABB();

#pragma omp parallel for
    for (uint32_t s = 0; s < baseSize.x; ++s) {
        resStrandVertCount[s] = std::max((baseStrandVertsCount[s] - 1), 0u) * mul + 1;
#pragma omp parallel for
        for (uint32_t v = 0; v < baseStrandVertsCount[s] - 1; ++v) {
            auto bIniVert = baseInitialVerts->GetEntryVal(s, v);  // base initial vert.
            auto bCurVert = baseCurrentVerts->GetEntryVal(s, v);  // base current vert.
            auto bCurVel = baseCurrentVels->GetEntryVal(s, v);    // base current vel.

            auto nbIniVert = baseInitialVerts->GetEntryVal(s, v + 1);  // next base initial vert.
            auto nbCurVert = baseCurrentVerts->GetEntryVal(s, v + 1);  // next base current vert.
            auto nbCurVel = baseCurrentVels->GetEntryVal(s, v + 1);    // next base current vel.

            for (uint32_t i = 0; i < mul; ++i) {
                // if (s == 0 && v == 0 && i == 0) {
                //     printf("%f -> %f -> %f -> %f\n\n", interpolate(bIniVert, nbIniVert, mul, 0).x,  // 0/3 = current
                //            interpolate(bIniVert, nbIniVert, mul, 1).x,                              // 1/3
                //            interpolate(bIniVert, nbIniVert, mul, 2).x,                              // 2/3
                //            interpolate(bIniVert, nbIniVert, mul, 3).x);                             // 3/3 = next
                // }
                resInitialVerts->SetEntryToArray(interpolate(bIniVert, nbIniVert, mul, i), s, v * mul + i);
                resCurrentVerts->SetEntryToArray(interpolate(bCurVert, nbCurVert, mul, i), s, v * mul + i);
                resCurrentVels->SetEntryToArray(interpolate(bCurVel, nbCurVel, mul, i), s, v * mul + i);
            }

            auto bfIniVert = baseInitialVerts->GetEntryVal(s, baseStrandVertsCount[s] - 1);
            auto bfCurVert = baseCurrentVerts->GetEntryVal(s, baseStrandVertsCount[s] - 1);
            auto bfCurVel = baseCurrentVels->GetEntryVal(s, baseStrandVertsCount[s] - 1);

            resInitialVerts->SetEntryToArray(bfIniVert, s, (baseStrandVertsCount[s] - 1) * mul);
            resCurrentVerts->SetEntryToArray(bfCurVert, s, (baseStrandVertsCount[s] - 1) * mul);
            resCurrentVels->SetEntryToArray(bfCurVel, s, (baseStrandVertsCount[s] - 1) * mul);
        }
    }

    for (uint32_t s = 0; s < baseSize.x; ++s) {
        *resVertCount += resStrandVertCount[s];
    }

    *hair = std::move(*resHair);
}