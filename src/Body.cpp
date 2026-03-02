#include "Body.h"

#include "../extern/glm/glm/gtc/matrix_transform.hpp"

#include "utilities/Geometry.h"
#include "utilities/FileUtil.h"
#include "utilities/Notification.h"
#include "exporters/ObjExporter.h"

const glm::uvec3 Body::HEAD_SDF_GRID_SIZE = glm::uvec3(64, 128, 64);

using Kami::Notification::MakeRedString;

void Body::ResetObj(string filePath, glm::vec3 pos, glm::vec3 scale) {
    confirmed = false;
    SetObj(filePath, pos, scale);
    GenerateSDF();
    Confirm();
}

void Body::SetObj(string filePath, glm::vec3 pos, glm::vec3 scale) {
    if (confirmed) {
        Kami::Notification::Warn(__func__, "Changing the head model is denied because this Body has already been confirmed.");
    }

    if (type != HEAD_ONLY && type != BOX) {
        Kami::Notification::Warn(__func__, "This instance hasn't been set as BodyType::HEAD_ONLY or BodyType::BOX.");
    }

    // if (initialHeadOrBoxPNMesh) {
    //     Kami::Notification::Caution(__func__, "The head model is going to be redefined.");
    // }
    initialHeadOrBoxPNMesh.reset(new PNMesh(filePath, pos, scale));
}

void Body::GenerateSDF(glm::uvec3 SDFGridSize) {
    if (confirmed) {
        Kami::Notification::Warn(__func__, "(re)GeneratingSDF is denied because this Body has already been confirmed.");
    }

    if (type == HEAD_ONLY || type == BOX) {
        if (!initialHeadOrBoxPNMesh) {
            Kami::Notification::Warn(__func__,
                                     "The head model hasn't been set.\n Please call " + MakeRedString("SetHead()") + "before calling this.");
        }
        initialHeadOrBoxPNMesh->GenerateSDF(SDFGridSize);
    } else if (type == FULLBODY) {
        //        currentFullbodyPNMesh->GenerateSDF(aabb, SDFGridSize);
    }
}

sh_ptr<PNMesh> Body::GetInitialHeadOrBoxPNMeshPtr() {
    if (!initialFullbodyPNMesh) {
        Kami::Notification::Warn(__func__, "The head model hasn't been set.\n Please call " + MakeRedString("SetHead()") + "before calling this.");
    }
    return initialHeadOrBoxPNMesh;
}

sh_ptr<PNMesh> Body::GetCurrentHeadOrBoxPNMeshPtr() {
    if (!currentHeadOrBoxPNMesh) {
        Kami::Notification::Warn(__func__, "The head hasn't been confirmed.\n Please call " + MakeRedString("Confirm()") + "before calling this.");
    }
    return currentHeadOrBoxPNMesh;
}

void Body::Confirm() {
    if (type == HEAD_ONLY || type == BOX) {
        if (!initialHeadOrBoxPNMesh) {
            Kami::Notification::Warn(__func__,
                                     "The body model hasn't been set.\n Please call " + MakeRedString("SetHead()") + "before calling this.");
        }
        currentHeadOrBoxPNMesh.reset(new PNMesh(*initialHeadOrBoxPNMesh));
    }

    // else if (type == FULLBODY && !currentFullbodyPNMesh) {
    //      Kami::Notification::Warn(__func__,
    //                               "The head model hasn't been set.\n Please call " + MakeRedString("SetFullbody()") + "before calling this.");
    //  }

    confirmed = true;
}
