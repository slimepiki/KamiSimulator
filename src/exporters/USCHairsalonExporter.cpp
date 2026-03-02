#include "USCHairsalonExporter.h"

#include "utilities/FileUtil.h"
#include "utilities/Notification.h"

void ExportUSCHairCore(string dstFileName, glm::uvec2 hairSize, unq_ptr<Kami::LinearUtil::EigenVecX3f2DArray>& vertices,
                       unq_ptr<uint32_t[]>& strandVertCount) {
    unq_ptr<KamiFile> file(make_unique<KamiFile>(dstFileName + ".data", std::ios_base::binary | std::ios_base::out | std::ios_base::trunc));
    file->WriteToBinary(&hairSize.x, sizeof(uint32_t));

    for (uint32_t i = 0; i < hairSize.x; i++) {
        file->WriteToBinary(&strandVertCount[i], sizeof(uint32_t));
        for (uint32_t j = 0; j < strandVertCount[i]; j++) {
            glm::vec3 val = vertices->GetEntryVal(i, j);
            file->WriteToBinary(&val, sizeof(glm::vec3));
        }
    }
}  // the file is closed automatically.

void ExportHairAsUSCHairsalon(string dstFileName, sh_ptr<Hair> hair) {
    auto hairSize = hair->GetHairParams().hairSize;
    auto& vertices = hair->GetCurrentVerticesRef();
    auto& strandVertCount = hair->GetStrandVertCountRef();

    ExportUSCHairCore(dstFileName, hairSize, vertices, strandVertCount);
}

void GenerateHorizontalStrandsArrayAsUSCHairsalon(string dstFileName, uint32_t numStrands, uint32_t numVertsPerStrand, glm::vec3 origin,
                                                  float hairLength, float interval) {
    glm::uvec2 hairSize = glm::uvec2(numStrands, numVertsPerStrand);
    unq_ptr<Kami::LinearUtil::EigenVecX3f2DArray> vertices = make_unique<Kami::LinearUtil::EigenVecX3f2DArray>(hairSize);
    unq_ptr<uint32_t[]> strandVertCount = make_unique<uint32_t[]>(numStrands);

    for (uint32_t i = 0; i < numStrands; ++i) {
        strandVertCount[i] = numVertsPerStrand;
        for (uint32_t j = 0; j < numVertsPerStrand; ++j) {
            float x = origin.x + j * hairLength / (float)(numVertsPerStrand - 1);
            float y = origin.y + 0;
            float z = origin.z + interval * i;
            glm::vec3 v = glm::vec3(x, y, z);
            vertices->SetEntryToArray(v, i, j);
        }
    }

    ExportUSCHairCore(dstFileName, hairSize, vertices, strandVertCount);
}
