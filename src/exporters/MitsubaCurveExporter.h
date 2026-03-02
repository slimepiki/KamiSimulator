#ifndef MITSUBA_BSPLINE_EXPORTER_H_
#define MITSUBA_BSPLINE_EXPORTER_H_

#include "Kami.h"
#include "utilities/LinearUtil.h"
#include "../utilities/FileUtil.h"
#include "../Hair.h"

#define HAIR_RADII 0.0008

// Please do not include any extension in dstFileName.
void ExportMitsubaCurve(string dstFileName, wk_ptr<Hair> hair, float radii = HAIR_RADII);

#endif /* MITSUBA_BSPLINE_EXPORTER_H_ */