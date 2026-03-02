#include "Kami.h"

#include "utilities/Notification.h"
#include "utilities/LinearUtil.h"
#include "utilities/JsonParser.h"
#include "KamiSimulator.h"

int main(int argc, char* argv[]) {
    // Please handle HairBodyMotion through shared_ptr.
    // Otherwise, the bad_weak_ptr exception occurs at shared_from_this in KamiSimulator::PickSolver.
    sh_ptr<KamiSimulator> kami = make_shared<KamiSimulator>();

    if (argc == 1) {  // manual setting
        string hairFilePath = "../resources/hairstyles/strands00001.data";
        string headFilePath = "../resources/hairstyles/head_model.obj";

        // enable the log file
        kami->SetLogFile();

        kami->TransplantHair(hairFilePath);

        // // for normal hair sim
        kami->Embody(headFilePath, Body::BodyType::HEAD_ONLY);
        //// for cantilever test
        // kami->Embody();

        //  set environment's params
        kami->SetSimulationTimings(2, 60, 5);

        // set output type (Video only, model and video e.t.c.)
        kami->SetOutputType(Kami::OutputType::VIDEO_ONLY);

        // Chose a hair model (e.g. Dummy model (ModelType::TEST))
        kami->PickSolver(Solver::ModelType::STABLE_COSSERAT_RODS);

        // Check wether the environment is valid or not and make the environment unchanged
        kami->ConfirmSimulationEnvironment();
    } else {  // json setting
        string jsonPath(argv[1]);
        kami->SetEnvironmentFromJson(jsonPath);
    }

    kami->Simulate();
    return 0;
}