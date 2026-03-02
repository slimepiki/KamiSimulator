#include "Kami.h"

#include "utilities/Notification.h"
#include "utilities/LinearUtil.h"
#include "KamiSimulator.h"

int main(int args, char* argv[]) {
    string hairFilePath = "../resources/hairstyles/strands00001.data";
    string headFilePath = "../resources/hairstyles/head_model.obj";

    // Please handle HairBodyMotion through shared_ptr.
    // Otherwise, the bad_weak_ptr exception occurs at shared_from_this in KamiSimulator::PickSolver.
    sh_ptr<KamiSimulator> kami = make_shared<KamiSimulator>("test");

    // enable the log file
    kami->SetLogFile();

    // load a hair .data
    kami->TransplantHair(hairFilePath);
    // kami->TransplantHair(hairFilePath, 1000, 100);

    kami->Embody(headFilePath, Body::BodyType::HEAD_ONLY);
    // set environment's params
    kami->SetSimulationTimings(2, 60, 5);

    // set output type (Video only, model and video e.t.c.)
    kami->SetOutputType(Kami::OutputType::VIDEO_ONLY);

    // Chose a hair model (e.g. Dummy model (ModelType::TEST))
    kami->PickSolver(Solver::ModelType::AUG_MASS_SPRING);

    // Check wether the environment is valid or not and make the environment unchanged
    // kami->ConfirmSimulationEnvironment();
    kami->ConfirmSimulationEnvironment();

    // for SDF test
    // kami->SaveBodySDF();

    kami->Simulate();
    return 0;
}