#ifndef Kami_H_
#define Kami_H_

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../extern/glm/glm/glm.hpp"

template <class T>
using sh_ptr = std::shared_ptr<T>;
template <class T>
using unq_ptr = std::unique_ptr<T>;
template <class T>
using wk_ptr = std::weak_ptr<T>;
using std::array;
using std::cerr;
using std::cout;
using std::endl;
using std::make_shared;
using std::make_unique;
using std::move;
using std::string;
using std::to_string;
using std::vector;

#define DEBUG_MODE
#define AUG_MASS_TEST

namespace Kami {

enum OutputType {
    // this will cause undifined behavior
    UNDEF_OUTPUT,
    // just simulating and save nothing
    NOTHING,
    // outputs sequential hairs' .datas and body's .objs
    SEQUENTIAL_OBJ_AND_DATA,
    // outputs a video and deletes hairs' .datas and body's .ojbs after creating vedeo
    VIDEO_ONLY,
    // outputs a video, and sequential hairs' .datas and body's .objs
    VIDEO_AND_OBJ
};
struct KamiSimulatorParams {
    // unique ID (This can be used at standalone solvers to avoid unnecessary initialization)
    // uint32_t SimulatorID;

    // I/O params
    // DO NOT include the extension.
    string OutputNamePrefix;

    OutputType outputType;

    // enum from Solver::ModelType
    int modelType;

    // timigs
    float simulationLengthSec;
    uint32_t drawFramePerSec;
    uint32_t numOfInternalStepPerDrawFrame;

    // translation of the body and the hair
    glm::vec3 globalPos;
    KamiSimulatorParams(KamiSimulatorParams& _params) {
        OutputNamePrefix = _params.OutputNamePrefix;
        outputType = _params.outputType;
        modelType = _params.modelType;
        simulationLengthSec = _params.simulationLengthSec;
        drawFramePerSec = _params.drawFramePerSec;
        numOfInternalStepPerDrawFrame = _params.numOfInternalStepPerDrawFrame;
    }

    KamiSimulatorParams() {
        OutputNamePrefix = "";
        outputType = OutputType::UNDEF_OUTPUT;
        modelType = 0;
        simulationLengthSec = 0;
        drawFramePerSec = 0;
        numOfInternalStepPerDrawFrame = 0;
    }
};
}  // namespace Kami

#endif /* Kami_H_ */
