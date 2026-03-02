#include "ObjExporter.h"
#include "USCHairsalonExporter.h"

#include "utilities/FileUtil.h"
#include "utilities/Notification.h"

void ExportPNMeshToObj(string dstFileName, sh_ptr<PNMesh> mesh) {
    unq_ptr<KamiFile> kamiFile(make_unique<KamiFile>(dstFileName + ".obj", std::ios_base::out | std::ios_base::trunc));

    auto& positions = mesh->GetPositionsRef();
    auto& normals = mesh->GetNormalsRef();
    auto& triangleIndices = mesh->GetTriangleIndicesRef();

    auto fst = kamiFile->GetFstream();

    auto meshinfo = mesh->GetPNMeshParam();

    //// Header
    int a = dstFileName.find_last_of('/');
    int b = dstFileName.find_last_of('.');
    string name = dstFileName.substr(a + 1, b - a - 1);

    (*fst) << "o " << name << endl;

    // Vertices
    for (uint32_t i = 0; i < meshinfo.numVertices; i++) {
        auto pos = glm::vec3(meshinfo.masterTrans * glm::vec4(positions[i], 1.0));
        (*fst) << "v " << pos.x << " " << pos.y << " " << pos.z << endl;
    }

    // normals
    for (uint32_t i = 0; i < meshinfo.numVertices; i++) {
        auto normal = glm::vec3(meshinfo.masterTrans * glm::vec4(normals[i], 1.0));
        (*fst) << "vn " << normal.x << " " << normal.y << " " << normal.z << endl;
    }

    // Indices
    for (uint32_t i = 0; i < meshinfo.numFaces; i++) {
        auto& tri = triangleIndices[i];
        (*fst) << "f " << tri.x + 1 << "//" << tri.x + 1 << " " << tri.y + 1 << "//" << tri.y + 1 << " " << tri.z + 1 << "//" << tri.z + 1 << endl;
    }
}  // the file is closed automatically.

void ExportBodyToObj(string dstFileName, sh_ptr<Body> body) {
    auto type = body->GetBodyType();
    if (type == Body::BodyType::HEAD_ONLY || type == Body::BodyType::BOX) {
        ExportPNMeshToObj(dstFileName, body->GetCurrentHeadOrBoxPNMeshPtr());
    } else if (type == Body::BodyType::FULLBODY) {
    }
}
