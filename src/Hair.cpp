#include "Hair.h"

#include "utilities/FileUtil.h"
#include "utilities/Notification.h"
#include <algorithm>

void Hair::CalcAABB() {
    if (!initialVertices) {
        Kami::Notification::Warn(__func__, "The hair hasn't been loaded yet. Please call any loader before calling CalcAABB().");
    } else {
        Kami::AABB tempAABB;
        for (uint32_t i = 0; i < params.hairSize.x; i++) {
            for (uint32_t j = 0; j < params.hairSize.y; j++) {
                auto hairVert = initialVertices->GetEntryVal(i, j);
                tempAABB.Update(hairVert);
            }
        }
        *aabb = tempAABB;
    }
}

void Hair::Resize(glm::uvec2 newSize) {
    if (!initialVertices)
        initialVertices = make_unique<EigenVecX3f2DArray>(newSize);
    else
        initialVertices->Resize(newSize);

    if (!currentVertices)
        currentVertices = make_unique<EigenVecX3f2DArray>(newSize);
    else
        currentVertices->Resize(newSize);

    if (!currentVelocities)
        currentVelocities = make_unique<EigenVecX3f2DArray>(newSize);
    else
        currentVelocities->Resize(newSize);

    unq_ptr<uint32_t[]> newVertCount = make_unique<uint32_t[]>(newSize.x);

    for (uint32_t i = 0; i < newSize.x; i++) {
        newVertCount[i] = 0;
    }

    uint32_t copysize = std::min(params.hairSize.x, newSize.x);
    std::copy(strandVertCount.get(), strandVertCount.get() + copysize, newVertCount.get());
    strandVertCount = std::move(newVertCount);

    params.hairSize = newSize;
}

unq_ptr<EigenVecX3f2DArray>& Hair::GetInitialVerticesRef() { return initialVertices; }

unq_ptr<EigenVecX3f2DArray>& Hair::GetCurrentVerticesRef() { return currentVertices; }

unq_ptr<EigenVecX3f2DArray>& Hair::GetCurrentVelocitiesRef() { return currentVelocities; }

unq_ptr<uint32_t[]>& Hair::GetStrandVertCountRef() { return strandVertCount; }

Kami::AABB Hair::GetAABB() {
    if (!aabb) {
        CalcAABB();
    }
    return *aabb;
}

Hair& Hair::operator=(Hair&& other) & noexcept {
    initialVertices = std::move(other.initialVertices);
    currentVertices = std::move(other.currentVertices);
    currentVelocities = std::move(other.currentVelocities);
    strandVertCount = std::move(other.strandVertCount);
    aabb = std::move(other.aabb);
    verticesCount = std::move(other.verticesCount);
    params = other.params;
    return *this;
}
