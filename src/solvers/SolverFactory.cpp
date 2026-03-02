#include "SolverFactory.h"
#include "AugMassSpring/AugMassSpringSolver.h"
#include "StableCosseratRods/StableCosseratRodsSolver.h"
#include "SeparatedAUGMSForTest.h"
#include "SDFCollisionSolver/SDFBodyCollisionSolver.h"
#include "SelleSolver.h"
#include "CantileverTestSolver.h"
#include "DummySolver.h"
#include "../utilities/Notification.h"

sh_ptr<Solver> SolverFactory::CreateSolver(Solver::ModelType type, sh_ptr<KamiSimulator> kami) {
    if (type == Solver::ModelType::AUG_MASS_SPRING) {
        return make_shared<AugMassSpringSolver>(kami);
    } /*else if (type == Solver::ModelType::SELLE_MASS_SPRING) {
       return make_shared<SelleSolver>(kami);
   } */
    else if (type == Solver::ModelType::SEP_AUGMS_FOR_TEST) {
        return make_shared<SeparatedAUGMSForTest>(kami);
    } else if (type == Solver::ModelType::CANTILEVER) {
        return make_shared<CantileverTestSolver>(kami);
    } else if (type == Solver::ModelType::STABLE_COSSERAT_RODS) {
        return make_shared<StableCosseratRodsSolver>(kami);
    } else {
        Kami::Notification::Warn(__func__, "Invalid type.");
    }
    return make_shared<DummySolver>(kami);
}