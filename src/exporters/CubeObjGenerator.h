#ifndef CUBE_OBJ_GENERATOR_H_
#define CUBE_OBJ_GENERATOR_H_

#include <string>
#include "../../extern/glm/glm/common.hpp"

// Please do not include any extension in dstFileName.
// min: The coordinate of the minimum corner of the cube.
// max: The coordinate of the maximam corner of the cube.
void GenerateCubeObjFile(std::string dstFileName, glm::vec3 min, glm::vec3 max);

#endif /* CUBE_OBJ_GENERATOR_H_ */
