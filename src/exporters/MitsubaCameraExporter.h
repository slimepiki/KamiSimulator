#ifndef MITSUBA_CAMERA_EXPORTER_H_
#define MITSUBA_CAMERA_EXPORTER_H_

#include "Kami.h"
#include <string>
#include "../../extern/glm/glm/common.hpp"
#include "../Camera.h"

// Please do not include any extension in dstFileName.
void ExporteMitsubaCameraTSV(string dstFileName, Camera::Setting cameraSetting);

#endif /* MITSUBA_CAMERA_EXPORTER_H_ */