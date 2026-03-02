# Cantilever Test

Kami simulator has a cantilever test function.

If you want to test a new solver, you should do at least,

1. Modify ```CantileverTestSolver::Construct(string settingJsonPath)``` around the comment which says ```\\ Set simulation model``` to initialize the new solver.
2. Modify ```CantileverTestSolver::Construct(string settingJsonPath)``` around the comment which says ```\\ You should add the new key if you want to test another parameter and add the corresponding line below.``` and add the corresponding parameter to ```struct CantileverTestSolver::CantileverJsonParams``` to add the parameters you want to test.

The common JSON parameters in this test are ```NumSolvers``` and ```SolverModels``` only. Refer [CantilSetting.json](../scripts/json/CantilSetting.json) and [KamiSettingForCantil.json](../scripts/json/KamiSettingForCantil.json) for the more datail.

You may have to modify ```YourSolver::Construct()``` to propagate ```SubstituteHair()``` and ```SubstituteBody()``` if the solver has a nested solver (See [StableCosseratRodsSolver::Construct()](../src/solvers/StableCosseratRods/StableCosseratRodsSolver.cpp)).
