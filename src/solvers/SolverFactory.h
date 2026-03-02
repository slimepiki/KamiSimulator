#ifndef SOLVER_FACTORY_H_
#define SOLVER_FACTORY_H_

#include "Solver.h"
#include "../KamiSimulator.h"

class KamiSimulator;

class SolverFactory {
   public:
    static sh_ptr<Solver> CreateSolver(Solver::ModelType type, sh_ptr<KamiSimulator> kami);
};

#endif /* SOLVER_FACTORY_H_ */