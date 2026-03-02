#ifndef USCHAIRSALON_EXPORTER_H_
#define USCHAIRSALON_EXPORTER_H_

#include "Kami.h"
#include "utilities/LinearUtil.h"
#include "../Hair.h"

using Kami::LinearUtil::EigenVecX3f2DArray;

//(hair length, hair interval)
constexpr glm::vec2 DEFAULT_HORIZONTAL_HAIR_GEN_PARAM = glm::vec2(0.25, 0.02);

// USCHairsalon-like Hair data exporter
// https://huliwenkidkid.github.io/liwenhu.github.io/
// If a file with the same name already exists, the original file will be deleted.
// Please do not include any extension in dstFileName.
void ExportHairAsUSCHairsalon(string dstFileName, sh_ptr<Hair> hair);

// Generate horizontally straight Hairs in a row
void GenerateHorizontalStrandsArrayAsUSCHairsalon(string dstFileName, uint32_t numStrands, uint32_t numVertsPerStrand,
                                                  glm::vec3 origin = glm::vec3(0, 0, 0), float hairLength = DEFAULT_HORIZONTAL_HAIR_GEN_PARAM.x,
                                                  float interval = DEFAULT_HORIZONTAL_HAIR_GEN_PARAM.y);
#endif /* USCHAIRSALON_EXPORTER_H_ */