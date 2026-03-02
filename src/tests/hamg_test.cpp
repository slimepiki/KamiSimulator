#include "Kami.h"
#include "utilities/Notification.h"
#include "utilities/LinearUtil.h"
#include "Hair.h"
#include "KamiSimulator.h"

int main(int args, char* argv[]) {
    string filepath = "../resources/hairstyles/strands00001.data";

    // Please handle HairBodyMotion through shared_ptr.
    // Otherwise, the bad_weak_ptr exception occurs at shared_from_this in KamiSimulator::PickSolver.
    sh_ptr<KamiSimulator> kami = make_shared<KamiSimulator>("test");

    // kami->SetLogFile();
    // kami->TransplantHair(filepath);
    // kami->SetSimulationTimings(0.2f, 0.05f, 2.0f);
    // kami->PickSolver(Solver::ModelType::TEST);
    // kami->ForcedConfirmSimulationEnvironment();

    // auto hair = kami->GetHair();
    // for (int i = 0; i < 20; i++) {
    //     Kami::Notification::PrintHairVertex2D(*hair, 0u, i);
    // }

    // kami->Simulate(false);
    return 0;
}