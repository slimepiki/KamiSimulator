#ifndef DUMMY_SOLVER_H_
#define DUMMY_SOLVER_H_

#include "../Kami.h"

#include "Solver.h"
#include <thread>
#include <chrono>

class DummySolver : public Solver {
   public:
    DummySolver(wk_ptr<KamiSimulator> _kami) : Solver(_kami) {};
    void Construct(string settingJsonPath = "") override {};
    void SolveNextStep(float dt) override { std::this_thread::sleep_for(std::chrono::milliseconds((int)(dt * 1000))); };
    void UploadHair() override {};
    void DownloadHair() override {};
};

#endif /* DUMMY_SOLVER_H_ */