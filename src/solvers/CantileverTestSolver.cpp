#include "CantileverTestSolver.h"
#include "SolverFactory.h"
#include "../utilities/FileUtil.h"
#include "../exporters/USCHairsalonExporter.h"
#include "../exporters/CubeObjGenerator.h"
#include "../importers/USCHairsalonImporter.h"

CantileverTestSolver::CantileverParams::CantileverParams(uint32_t _numSolvers, uint32_t _numVertsPerStrands, float _strandLength, float _hairInterval,
                                                         glm::vec3 _firstHairOrigin)
    : numSolvers(_numSolvers),
      numVertsPerStrands(_numVertsPerStrands),
      strandLength(_strandLength),
      hairInterval(_hairInterval),
      firstHairOrigin(_firstHairOrigin) {}

CantileverTestSolver::CantileverTestSolver(wk_ptr<KamiSimulator> _kami) : Solver(_kami) { params = CantileverParams::GetDefaultCantileverParams(); }

void CantileverTestSolver::Construct(string settingJsonPath) {
    CantileverJsonParams cjparams{};
    bool isDefault = true;

    if (Kami::FileUtil::IsFileExist(settingJsonPath) && LoadSettingJson(settingJsonPath, cjparams)) {
        isDefault = false;
    }
    params.numSolvers = cjparams.NumSolvers;

    // hair (re)construction
    GenerateHorizontalStrandsArrayAsUSCHairsalon(GenerateKamiFilename("augHoriHair"), params.numSolvers, params.numVertsPerStrands,
                                                 params.firstHairOrigin, params.strandLength, params.hairInterval);
    ImportUSCHairSalonToHair(GenerateKamiFilename("augHoriHair") + ".data", GetHair(), 10000, 100);

    auto defBox = Body::DEFAULT_BOX_HALFWIDTH;
    // body (re)construction
    glm::vec3 min{-defBox}, max{glm::vec3(0, defBox.y, defBox.z * (params.numSolvers))};

    GenerateCubeObjFile(GenerateKamiFilename("canti_box"), min, max);
    GenerateHorizontalStrandsArrayAsUSCHairsalon(GenerateKamiFilename("single_strand"), 1, 100);
    GetBody().lock()->ResetObj(GenerateKamiFilename("canti_box") + ".obj");

    JsonParser jsonchecker(settingJsonPath);
    auto solverParams = GetKamiParams();
    sh_ptr<KamiSimulator> dummykami = make_shared<KamiSimulator>(solverParams);
    dummykami->CreateDummy(GetHair(), GetBody(), solverParams);

    if (isDefault) {  // This branch is nonsense.
        for (uint32_t i = 0; i < Solver::MODEL_TYPE_COUNT; ++i) {
            if (i == Solver::ModelType::UNDEF_MODEL || i == Solver::ModelType::CANTILEVER) continue;
            // ########Strand preparation (Common part)########
            sh_ptr<Hair> strand = make_shared<Hair>();
            strands.push_back(strand);
            ImportUSCHairSalonToHair(GenerateKamiFilename("single_strand") + ".data", strand, 10000, 100);
            // ##################################

            sh_ptr<Solver> newSolver = SolverFactory::CreateSolver((Solver::ModelType)i, dummykami);
            newSolver->SubstituteHair(strand);
            newSolver->Construct();
            solvers.push_back(newSolver);
        }
    } else {
        for (uint32_t i = 0; i < cjparams.NumSolvers; ++i) {
            // ########Strand preparation (Common part)########
            sh_ptr<Hair> strand = make_shared<Hair>();
            strands.push_back(strand);
            ImportUSCHairSalonToHair(GenerateKamiFilename("single_strand") + ".data", strand, 10000, 100);
            // ##################################

            sh_ptr<Solver> newSolver;

            // Set simulation model
            auto smodel = cjparams.SolverModels[i];
            if (smodel == "AUGMS") {
                newSolver = SolverFactory::CreateSolver(Solver::ModelType::AUG_MASS_SPRING, dummykami);
            } else if (smodel == "STCOSSERAT") {
                newSolver = SolverFactory::CreateSolver(Solver::ModelType::STABLE_COSSERAT_RODS, dummykami);
            } else {
                Kami::Notification::Warn(__func__, Kami::Notification::GetNumberString(i) + " model type is invalid. Please Check its spelling.");
            }

            // substituting. You may have to substitute the sub-solver's sub-solver hair recursively.
            newSolver->SubstituteHair(strand);

            // Create Json for Solvers' setting
            JsonParser jp;

            // You should add the new key if you want to test another parameter and add the corresponding line below.
            string tempJsonPath = GenerateKamiFilename("temp_json") + ".json";
            if (jsonchecker.CheckKey("WindPeaks")) jp.SetVal("WindPeak", (int)cjparams.WindPeaks[i]);
            if (jsonchecker.CheckKey("BendKs")) jp.SetVal("BendK", cjparams.BendKs[i]);
            if (jsonchecker.CheckKey("StretchKs")) jp.SetVal("StretchK", cjparams.StretchKs[i]);
            if (jsonchecker.CheckKey("Young")) jp.SetVal("Young", cjparams.Youngs[i]);
            if (jsonchecker.CheckKey("ResampleLens")) jp.SetVal("ResampleLen", cjparams.ResampleLens[i]);
            if (jsonchecker.CheckKey("UseYoungsModule")) jp.SetVal("UseYoungs", cjparams.UseYoungs[i]);
            if (jsonchecker.CheckKey("SCREnableSDF")) jp.SetVal("EnableSDF", cjparams.SCREnableSDF[i]);
            if (jsonchecker.CheckKey("SDFThreshold")) jp.SetVal("SDFThreshold", 0.013f);

            jp.SaveJson(tempJsonPath, true);

            newSolver->Construct(tempJsonPath);
            solvers.push_back(newSolver);
            Kami::FileUtil::DeleteFile(tempJsonPath);
        }
    }
    Kami::FileUtil::DeleteFile(GenerateKamiFilename("augHoriHair") + ".data");
}

void CantileverTestSolver::SolveNextStep(float dt) {
    // solving each strand
    for (auto solver : solvers) {
        solver->SolveNextStep(dt);
    }
}

CantileverTestSolver::~CantileverTestSolver() {
    Kami::FileUtil::DeleteFile(GenerateKamiFilename("single_strand") + ".data");
    Kami::FileUtil::DeleteFile(GenerateKamiFilename("canti_box") + ".obj");
}

bool CantileverTestSolver::LoadSettingJson(string jsonPath, CantileverJsonParams& jsonParams) {
    JsonParser jp(jsonPath);
    CantileverJsonParams tempParams;

    if (jp.CheckKey("NumSolvers")) {
        tempParams.NumSolvers = jp.GetInteger("NumSolvers");
    } else {
        Kami::Notification::Caution(__func__, "Invalid \"NumSolvers\"");
        return false;
    }

    if (jp.CheckKey("SolverModels")) {
        tempParams.SolverModels = jp.GetStringVec("SolverModels");
        if (tempParams.NumSolvers > tempParams.SolverModels.size()) {
            Kami::Notification::Warn(__func__,
                                     "Json's SolverModels.size() is less than NumSolvers. \n SolverModels.size() must be lager than NumSolvers.");
            return false;
        }
    } else {
        Kami::Notification::Caution(__func__, "Invalid \"SolverModels\"");
        return false;
    }

    if (jp.CheckKey("WindPeaks")) {
        tempParams.WindPeaks = jp.GetFloatVec("WindPeaks");
        if (tempParams.NumSolvers > tempParams.WindPeaks.size()) {
            Kami::Notification::Warn(__func__, "Json's WindPeaks.size() is less than NumSolvers. \n WindPeaks.size() must be lager than NumSolvers.");
            return false;
        }
    }

    if (jp.CheckKey("BendKs")) {
        tempParams.BendKs = jp.GetFloatVec("BendKs");
        if (tempParams.NumSolvers > tempParams.BendKs.size()) {
            Kami::Notification::Warn(__func__, "Json's BendKs.size() is less than NumSolvers. \n BendKs.size() must be lager than NumSolvers.");
            return false;
        }
    }

    if (jp.CheckKey("StretchKs")) {
        tempParams.StretchKs = jp.GetFloatVec("StretchKs");
        if (tempParams.NumSolvers > tempParams.StretchKs.size()) {
            Kami::Notification::Warn(__func__, "Json's StretchKs.size() is less than NumSolvers. \n StretchKs.size() must be >= NumSolvers.");
            return false;
        }
    }

    if (jp.CheckKey("Youngs")) {
        tempParams.Youngs = jp.GetFloatVec("Youngs");
        if (tempParams.NumSolvers > tempParams.Youngs.size()) {
            Kami::Notification::Warn(__func__, "Json's Youngs.size() is less than NumSolvers. \n Youngs.size() must be >= NumSolvers.");
            return false;
        }
    }

    if (jp.CheckKey("ResampleLens")) {
        tempParams.ResampleLens = jp.GetFloatVec("ResampleLens");
        if (tempParams.NumSolvers > tempParams.ResampleLens.size()) {
            Kami::Notification::Warn(__func__, "Json's ResampleLens.size() is less than NumSolvers. \n ResampleLens.size() must be >= NumSolvers.");
            return false;
        }
    }

    if (jp.CheckKey("UseYoungs")) {
        tempParams.SCREnableSDF = jp.GetBoolVec("UseYoungs");
        if (tempParams.NumSolvers > tempParams.UseYoungs.size()) {
            Kami::Notification::Warn(__func__, "Json's UseYoungs.size() is less than NumSolvers. \n UseYoungs.size() must be >= NumSolvers.");
            return false;
        }
    }

    if (jp.CheckKey("SCREnableSDF")) {
        tempParams.SCREnableSDF = jp.GetBoolVec("SCREnableSDF");
        if (tempParams.NumSolvers > tempParams.SCREnableSDF.size()) {
            Kami::Notification::Warn(__func__, "Json's SCREnableSDF.size() is less than NumSolvers. \n SCREnableSDF.size() must be >= NumSolvers.");
            return false;
        }
    }

    jsonParams.NumSolvers = tempParams.NumSolvers;
    jsonParams.SolverModels = tempParams.SolverModels;
    jsonParams.WindPeaks = tempParams.WindPeaks;
    jsonParams.StretchKs = tempParams.StretchKs;
    jsonParams.BendKs = tempParams.BendKs;
    jsonParams.ResampleLens = tempParams.ResampleLens;
    jsonParams.SCREnableSDF = tempParams.SCREnableSDF;

    return true;
}

void CantileverTestSolver::DownloadHair() {
    for (auto solver : solvers) {
        solver->DownloadHair();
    }

    auto hair = GetHair();
    auto& hairVerts = hair.lock()->GetCurrentVerticesRef();
    auto& hairVels = hair.lock()->GetCurrentVelocitiesRef();
    auto& strandVertCounts = hair.lock()->GetStrandVertCountRef();
#pragma omp parallel for
    for (uint32_t i = 0; i < strands.size(); ++i) {
        auto& verts = strands[i]->GetCurrentVerticesRef();
        auto& vels = strands[i]->GetCurrentVelocitiesRef();
        auto count = strands[i]->GetStrandVertCountRef()[0];
        strandVertCounts[i] = count;
#pragma omp parallel for
        for (uint32_t j = 0; j < count; ++j) {
            hairVerts->SetEntryToArray(verts->GetEntryVal(0, j) + glm::vec3(0, 0, params.hairInterval * i), i, j);
            hairVels->SetEntryToArray(vels->GetEntryVal(0, j), i, j);
        }
    }
}

void CantileverTestSolver::UploadHair() {}
