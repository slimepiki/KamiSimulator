#include "SeparatedAUGMSForTest.h"

#include "AugMassSpring/AugMassSpringSolver.h"
#include "SDFCollisionSolver/SDFBodyCollisionSolver.h"
#include "../KamiSimulator.h"

SeparatedAUGMSForTest::SeparatedAUGMSForTest(wk_ptr<KamiSimulator> _kami) : Solver(_kami) {
    augmsSolver.reset(new AugMassSpringSolver(_kami));
    sdfSolver.reset(new SDFBodyCollisionSolver(_kami));
}

void SeparatedAUGMSForTest::Construct(string settingJsonPath) {
    augmsSolver->Construct();
    sdfSolver->Construct();
    augmsSolver->UseSDFOrNot(false);
    sdfSolver->EnableStrainLimiting(true);
}

void SeparatedAUGMSForTest::SolveNextStep(float dt) {
    augmsSolver->SolveNextStep(dt);
    augmsSolver->DownloadHair();

    sdfSolver->UploadHair();
    sdfSolver->SolveNextStep(dt);
}

void SeparatedAUGMSForTest::UploadHair() { augmsSolver->UploadHair(); }

void SeparatedAUGMSForTest::DownloadHair() { sdfSolver->DownloadHair(); }
