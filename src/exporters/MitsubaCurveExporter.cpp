#include "MitsubaCurveExporter.h"
#include "../utilities/FileUtil.h"
#include "../utilities/Notification.h"
#include <cstdlib>
#include <sstream>
#include <iomanip>

using ::Kami::Notification::MakeRedString;

void ExportMitsubaCurve(string dstFileName, wk_ptr<Hair> hair, float radii) {
    unq_ptr<KamiFile> kamiFile(make_unique<KamiFile>(dstFileName + ".mcur", std::ios_base::out | std::ios_base::trunc));

    auto hairsize = hair.lock()->GetHairParams().hairSize;
    auto& vertices = hair.lock()->GetCurrentVerticesRef();
    auto& strandVertCount = hair.lock()->GetStrandVertCountRef();

    unq_ptr<std::string[]> stArr = make_unique<std::string[]>(hairsize.x);

#pragma omp parallel for
    for (uint32_t i = 0; i < hairsize.x; i++) {
        stArr[i].reserve(hairsize.y * (sizeof(char) * 50));
        if (strandVertCount[i] < 2) continue;
        for (uint32_t j = 0; j < strandVertCount[i]; j++) {
            std::stringstream ss;
            auto v = vertices->GetEntryVal(i, j);
            float curradii = radii * (1.0 - 1.0 * sqrt(j / strandVertCount[i]));
            ss << std::fixed << std::setprecision(5) << v.x << " " << v.y << " " << v.z << " " << curradii << endl;
            stArr[i] += ss.str();
        }
    }

    for (uint32_t i = 0; i < hairsize.x; i++) {
        kamiFile->WriteStringWithBreak(stArr[i]);
    }
}  // the file is closed automatically.
