#include "SDFParaviewExporter.h"
#include "../utilities/FileUtil.h"
#include "../../extern/glm/glm/common.hpp"
#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>

void ExportSDFAsParaViewTSV(string dstFileName, sh_ptr<PNMesh> mesh) {
    unq_ptr<KamiFile> file(make_unique<KamiFile>(dstFileName + ".csv", std::ios_base::binary | std::ios_base::out | std::ios_base::trunc));

    vector<float3> positions;
    vector<float> distances;
    vector<float3> normals;
    mesh->StoreSDFQueries(positions, distances, normals);

    file->WriteStringWithNoBreak("X,Y,Z,Distance,NormalX,NormalY,NormalZ\n");

    for (uint32_t i = 0; i < positions.size(); i++) {
        std::stringstream ssarr;
        auto& p = positions[i];
        auto& d = distances[i];
        auto& n = normals[i];

        ssarr << std::fixed << std::setprecision(5) << p.x << "," << p.y << "," << p.z << "," << d << "," << n.x << "," << n.y << "," << n.z << endl;
        file->WriteStringWithNoBreak(ssarr.str());
    }

    cout << "SDF was saved at " + dstFileName + ".csv" << endl;
}  // the file is closed automatically.
