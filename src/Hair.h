#ifndef HAIR_H_
#define HAIR_H_

#include "Kami.h"
#include "utilities/Geometry.h"
#include "utilities/LinearUtil.h"

using Kami::LinearUtil::EigenVecX3f2DArray;

struct Hair {
   public:
    struct HairParams {
        // uvec2(num of strand, max vertices of the strand)
        glm::uvec2 hairSize;
        // Please include the extension.
        string srcFilePath;
    };

   private:
    HairParams params;

    unq_ptr<EigenVecX3f2DArray> initialVertices;
    unq_ptr<EigenVecX3f2DArray> currentVertices;
    unq_ptr<EigenVecX3f2DArray> currentVelocities;
    unq_ptr<uint32_t[]> strandVertCount = nullptr;

    unq_ptr<Kami::AABB> aabb;
    unq_ptr<uint32_t> verticesCount = make_unique<uint32_t>(0);

    void CalcAABB();

   public:
    // Hair(string filePath, uint32_t maxStrandCount = 10000, uint32_t maxLength = 100);
    Hair() {};
    void Resize(glm::uvec2 newSize);  // isn't tested

    // The KamiSimulator class uses these pointer during simulation.
    // Please load hair model before getVertices.
    unq_ptr<EigenVecX3f2DArray>& GetInitialVerticesRef();
    unq_ptr<EigenVecX3f2DArray>& GetCurrentVerticesRef();
    unq_ptr<EigenVecX3f2DArray>& GetCurrentVelocitiesRef();
    unq_ptr<uint32_t[]>& GetStrandVertCountRef();
    HairParams& GetHairParamsRef() { return params; }
    HairParams GetHairParams() const { return params; }
    unq_ptr<uint32_t>& GetVerticesCountRef() { return verticesCount; }
    uint32_t GetVerticesCount() const { return *verticesCount; }
    unq_ptr<Kami::AABB>& GetAABBRef() { return aabb; };
    Kami::AABB GetAABB();

    Hair& operator=(Hair&& other) & noexcept;
    Hair& operator=(const Hair&) = delete;
};

#endif /* Hair_H_ */