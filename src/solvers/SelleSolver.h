#ifndef SELLE_SOLVER_H_
#define SELLE_SOLVER_H_

#include "Solver.h"

// A Mass Spring Model for Hair Simulation
//( https://physbam.stanford.edu/~fedkiw/papers/stanford2008-02.pdf )

class SelleSolver : public Solver {
   public:
    SelleSolver(wk_ptr<KamiSimulator> _kami) : Solver(_kami) {};
    void Construct(string settingJsonPath = "") override;
    void SolveNextStep(float dt) override;
};

#endif /* SELLE_SOLVER_H_ */