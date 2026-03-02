#ifndef SDFPARAVIEW_EXPORTER_H_
#define SDFPARAVIEW_EXPORTER_H_

#include "../PNMesh.h"

// SDF to CSV expoter for ParaView
// https://www.paraview.org/
// If a file with the same name already exists, the original file will be deleted.
// Please do not include any extension in dstFileName.
void ExportSDFAsParaViewTSV(string dstFileName, sh_ptr<PNMesh> mesh);

#endif /* SDFPARAVIEW_EXPORTER_H_ */