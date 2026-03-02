#ifndef BODY_H_
#define BODY_H_

#include <fstream>

#include "../extern/glm/glm/glm.hpp"

#include "Kami.h"
#include "PNMesh.h"
#include "exporters/USCHairsalonExporter.h"

class PNMesh;

class Body {
   public:
    enum BodyType { HEAD_ONLY, FULLBODY, BOX };

    static constexpr glm::vec3 DEFAULT_BOX_HALFWIDTH = glm::vec3(0.01, 0.05, DEFAULT_HORIZONTAL_HAIR_GEN_PARAM.y);

   private:
    static const glm::uvec3 HEAD_SDF_GRID_SIZE;

    bool confirmed = false;
    BodyType type = HEAD_ONLY;

    // Head mesh which vertices has position and normal.
    sh_ptr<PNMesh> initialHeadOrBoxPNMesh;
    sh_ptr<PNMesh> currentHeadOrBoxPNMesh;

    // BodyMesh
    sh_ptr<PNMesh> initialFullbodyPNMesh;
    //    unq_ptr<AnimMesh> currentFullbodyPNMesh;

    // bone

    // void RecreateSDF();

   public:
    Body(BodyType _type) : type(_type) {};
    // Please include the extension.
    void ResetObj(string filePath, glm::vec3 pos = glm::vec3(0, 0, 0), glm::vec3 scale = glm::vec3(1, 1, 1));
    // Please include the extension.
    void SetObj(string filePath, glm::vec3 pos = glm::vec3(0, 0, 0), glm::vec3 scale = glm::vec3(1, 1, 1));
    // void SetFullbody(string filePath, glm::vec3 pos = glm::vec3(0, 0, 0), glm::vec3 scale = glm::vec3(1, 1, 1));

    void GenerateSDF(glm::uvec3 SDFGridSize = Body::HEAD_SDF_GRID_SIZE);

    // We can call this if type is HEAD_ONLY for now
    sh_ptr<PNMesh> GetInitialHeadOrBoxPNMeshPtr();
    sh_ptr<PNMesh> GetCurrentHeadOrBoxPNMeshPtr();
    BodyType GetBodyType() const { return type; }

    // SetKeyFrame(keyFrame);
    // GetSDF();

    void Confirm();
};

#endif /* BODY_H_ */