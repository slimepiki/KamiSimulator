#include "CubeObjGenerator.h"
#include "../utilities/Geometry.h"
#include "../utilities/FileUtil.h"
#include <iomanip>

void GenerateCubeObjFile(std::string dstFileName, glm::vec3 min, glm::vec3 max) {
    unq_ptr<KamiFile> kamiFile(make_unique<KamiFile>(dstFileName + ".obj", std::ios_base::out | std::ios_base::trunc));

    auto vertIdx = Kami::CubeObjData::CUBE_IDX;
    auto norms = Kami::CubeObjData::CUBE_NORMALS;
    auto faces = Kami::CubeObjData::CUBE_FACES;

    glm::vec3 cubeMinMax[2] = {min, max};

    // verts
    for (uint32_t i = 0; i < Kami::CubeObjData::IDX_SIZE.x; ++i) {
        std::stringstream ss;
        float x = cubeMinMax[vertIdx[i][0]].x;
        float y = cubeMinMax[vertIdx[i][1]].y;
        float z = cubeMinMax[vertIdx[i][2]].z;
        ss << std::fixed << std::setprecision(5) << "v " << x << " " << y << " " << z << endl;
        kamiFile->WriteStringWithNoBreak(ss.str());
    }

    // a blank
    kamiFile->WriteStringWithBreak("");

    // normals
    for (uint32_t i = 0; i < Kami::CubeObjData::NORMALS_SIZE.x; ++i) {
        std::stringstream ss;
        auto norm = norms[i];

        ss << std::fixed << std::setprecision(5) << "vn " << norm[0] << " " << norm[1] << " " << norm[2] << endl;
        kamiFile->WriteStringWithNoBreak(ss.str());
    }

    // a blank
    kamiFile->WriteStringWithBreak("");

    // faces
    for (uint32_t i = 0; i < Kami::CubeObjData::FACES_SIZE.x; ++i) {
        std::stringstream ss;

        ss << std::fixed << std::setprecision(5) << "f ";

        for (uint32_t j = 0; j < Kami::CubeObjData::FACES_SIZE.y; ++j) {
            if (j != 0) ss << " ";
            ss << faces[i][j][0] << "//" << faces[i][j][1];
        }

        ss << endl;
        kamiFile->WriteStringWithNoBreak(ss.str());
    }

    // a blank
    kamiFile->WriteStringWithBreak("");

    // the file is closed automatically.
}