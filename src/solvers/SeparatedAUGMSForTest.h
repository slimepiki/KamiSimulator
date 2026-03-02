#ifndef SEPARATED_AUGMSFOR_TEST_H_
#define SEPARATED_AUGMSFOR_TEST_H_

#include "../Kami.h"
#include "Solver.h"

class KamiSimulator;
class AugMassSpringSolver;
class SDFBodyCollisionSolver;

// This is example of a nested solver. Unfortunately, the nested solvers will be time-consuming due to the data transfer between the CPU and the GPU.
class SeparatedAUGMSForTest : public Solver {
   public:
    SeparatedAUGMSForTest(wk_ptr<KamiSimulator> _kami);

    // Please precompute or generate auxiliary things here.
    void Construct(string settingJsonPath = "") override;

    // dt : step size [ms]
    // In most cases, Hair's currentPos and currentVert is used as interface between the solver and KamiSolver.
    void SolveNextStep(float dt) override;
    void UploadHair() override;
    void DownloadHair() override;

   private:
    unq_ptr<AugMassSpringSolver> augmsSolver = nullptr;
    unq_ptr<SDFBodyCollisionSolver> sdfSolver = nullptr;
};

#endif /* SEPARATED_AUGMSFOR_TEST_H_ */
