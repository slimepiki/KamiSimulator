# Quick imprementation Guide for a new Solver

1. Inherit Solver class from ```solvers/Solver.h```
2. Override ```Solver::Construct(string settingJsonPath)```, ```Solver::SolveNextStep(float dt)```,  ```Solver::UploadHair()```, and ```Solver::DownloadHair()```. The details of the functions are on [Solver Interfaces page](solverInterface.md).
3. Add a new entry to enum ```Solver::ModelType```.
4. Include the created solver in ```SolverFactory.cpp```
5. Modify ```KamiSimulator::PickSolver(int mode)```, ```SetEnvironmentFromJson(string jsonPath)```, and ```SolverFactory::CreateSolver(Solver::ModelType type, sh_ptr<KamiSimulator> hbm)``` by following the example in the function.
