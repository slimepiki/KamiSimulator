#ifndef OBJ_EXPORTER_H_
#define OBJ_EXPORTER_H_

#include "Kami.h"
#include "utilities/LinearUtil.h"
#include "../Body.h"

// Please do not include any extension in dstFileName.
void ExportPNMeshToObj(string dstFileName, sh_ptr<PNMesh> mesh);

// Please do not include any extension in dstFileName.
void ExportBodyToObj(string dstFileName, sh_ptr<Body> body);

#endif /* OBJ_EXPORTER_H_ */