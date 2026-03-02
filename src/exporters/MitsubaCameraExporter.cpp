#include "MitsubaCameraExporter.h"
#include "../utilities/FileUtil.h"
#include <sstream>

void ExporteMitsubaCameraTSV(string dstFileName, Camera::Setting cameraSetting) {
    unq_ptr<KamiFile> kamiFile(make_unique<KamiFile>(dstFileName + ".tsv", std::ios_base::out | std::ios_base::trunc));

    std::stringstream ss;

    auto o = cameraSetting.origin;
    auto l = cameraSetting.lookAt;
    auto u = cameraSetting.up;

    std::stringstream originss, lookatss, upss;
    originss << o.x << " " << o.y << " " << o.z << endl;
    lookatss << l.x << " " << l.y << " " << l.z << endl;
    upss << u.x << " " << u.y << " " << u.z << endl;

    kamiFile->WriteStringWithNoBreak(originss.str());
    kamiFile->WriteStringWithNoBreak(lookatss.str());
    kamiFile->WriteStringWithNoBreak(upss.str());
}  // the file is closed automatically.