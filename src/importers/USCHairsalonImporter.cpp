#include "USCHairsalonImporter.h"

#include "utilities/FileUtil.h"
#include "utilities/Notification.h"
#include "utilities/LinearUtil.h"
#include "utilities/Geometry.h"

#include <algorithm>

inline bool CheckShrinkenIndexOrNot(uint32_t query, uint32_t shrinkenSize, uint32_t sizeFromFile) {
    return (query != 0) ? (query % (sizeFromFile / shrinkenSize) == 0) : true;
};

// inline bool CheckShrinkenIndexOrNot(uint32_t query, uint32_t uMax, uint32_t vMax, uint32_t sizeFromFile) {
//     return CheckShrinkenIndexOrNot(query, uMax * vMax, sizeFromFile);
// };

class KamiFile;

// USCHairsalon-like Hair data importer

namespace {
void ImportUSCHairSalon(string filepath, unq_ptr<EigenVecX3f2DArray>& glm3dverts, unq_ptr<uint32_t[]>& strandVertCount, glm::uvec2& hairSize,
                        unq_ptr<Kami::AABB>& aabb, unq_ptr<uint32_t>& vertCount);
// If you don't want to load full hair, choose a smaller size than the
// USCHairSalon data size, which shrinks the loaded hair size.
void ShrinkenImportUSCHairSalon(string filepath, unq_ptr<EigenVecX3f2DArray>& glm3dverts, unq_ptr<uint32_t[]>& strandVertCount, uint32_t maxStrand,
                                uint32_t maxLength, glm::uvec2& hairSize, unq_ptr<Kami::AABB>& aabb, unq_ptr<uint32_t>& vertCount);
}  // namespace

void ImportUSCHairSalonToHair(string filepath, wk_ptr<Hair> hair, uint32_t maxStrandCount, uint32_t maxLength) {
    auto& hairParams = hair.lock()->GetHairParamsRef();
    auto& hairSize = hairParams.hairSize;
    auto& glm3dInitialVerts = hair.lock()->GetInitialVerticesRef();
    auto& strandVertCount = hair.lock()->GetStrandVertCountRef();
    auto& aabb = hair.lock()->GetAABBRef();
    auto& vertCount = hair.lock()->GetVerticesCountRef();

    aabb.reset(new Kami::AABB());

    hairParams.srcFilePath = filepath;
    string ext = Kami::FileUtil::GetFileExtension(filepath);
    if (ext == "data") {
        if (maxStrandCount != 10000 || maxLength != 100)
            ShrinkenImportUSCHairSalon(filepath, glm3dInitialVerts, strandVertCount, maxStrandCount, maxLength, hairSize, aabb, vertCount);
        else
            ImportUSCHairSalon(filepath, glm3dInitialVerts, strandVertCount, hairSize, aabb, vertCount);
    } else {
        throw std::runtime_error("The extention \"" + ext + "\" can't be read.\n");
    }
    auto& glm3dCurrentVelocities = hair.lock()->GetCurrentVelocitiesRef();
    auto& glm3dCurrentVerts = hair.lock()->GetCurrentVerticesRef();

    glm3dCurrentVelocities = make_unique<EigenVecX3f2DArray>(hairSize);
    glm3dCurrentVerts = make_unique<EigenVecX3f2DArray>(hairSize);
    *glm3dCurrentVerts = *glm3dInitialVerts;
}

namespace {
void ImportUSCHairSalon(string filepath, unq_ptr<EigenVecX3f2DArray>& glm3dverts, unq_ptr<uint32_t[]>& strandVertCount, glm::uvec2& hairSize,
                        unq_ptr<Kami::AABB>& aabb, unq_ptr<uint32_t>& vertCount) {
    hairSize.y = 100;
    sh_ptr<KamiFile> file(make_unique<KamiFile>(filepath, std::ios_base::binary | std::ios_base::in));

    file->ReadFromBinary(&hairSize.x, sizeof(uint32_t));

    strandVertCount = make_unique<uint32_t[]>(hairSize.x);
    glm3dverts = make_unique<EigenVecX3f2DArray>(hairSize);

    for (uint32_t i = 0; i < hairSize.x; i++) {
        vector<glm::vec3> strand;
        uint32_t length;

        file->ReadFromBinary(&length, sizeof(uint32_t));
        strand.resize(length);
        strandVertCount[i] = length;
        *vertCount += strandVertCount[i];
        file->ReadFromBinary(strand.data(), sizeof(glm::vec3) * length);
        for (uint32_t j = 0; j < length; j++) {
            glm3dverts->SetEntryToArray(strand[j], i, j);
            aabb->Update(strand[j]);
        }
    }
    // the file is closed automatically.
}

void ShrinkenImportUSCHairSalon(string filepath, unq_ptr<EigenVecX3f2DArray>& glm3dverts, unq_ptr<uint32_t[]>& strandVertCount, uint32_t maxStrand,
                                uint32_t maxLength, glm::uvec2& hairSize, unq_ptr<Kami::AABB>& aabb, unq_ptr<uint32_t>& vertCount) {
    sh_ptr<KamiFile> file(make_unique<KamiFile>(filepath, std::ios_base::binary | std::ios_base::in));
    uint32_t numStrands;
    file->ReadFromBinary(&numStrands, sizeof(uint32_t));

    hairSize.x = std::min(maxStrand, numStrands);
    hairSize.y = std::min(100u, maxLength);

    strandVertCount = make_unique<uint32_t[]>(hairSize.x);
    glm3dverts = make_unique<EigenVecX3f2DArray>(hairSize);

    uint32_t iden = numStrands / hairSize.x;
    uint32_t jden;
    uint32_t length, u = 0, v = 0;
    for (uint32_t i = 0; i < numStrands; i++) {
        vector<glm::vec3> strand;

        file->ReadFromBinary(&length, sizeof(uint32_t));
        strand.resize(length);

        file->ReadFromBinary(strand.data(), sizeof(glm::vec3) * length);
        if (CheckShrinkenIndexOrNot(i, hairSize.x, numStrands)) {
            u = i / iden;
            strandVertCount[u] = std::min(hairSize.y, length);
            *vertCount += strandVertCount[u];
            for (uint32_t j = 0; j < length; j++) {
                if (CheckShrinkenIndexOrNot(j, maxLength, length)) {
                    jden = (length > maxLength) ? (length / maxLength) : 1;
                    v = j / jden;
                    glm3dverts->SetEntryToArray(strand[j], u, v);
                    aabb->Update(strand[j]);
                }
            }
        }
    }
    // the file is closed automatically.
}

}  // namespace
