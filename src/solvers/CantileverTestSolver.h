#ifndef CANTILEVER_TEST_SOLVER_H_
#define CANTILEVER_TEST_SOLVER_H_

#include "Solver.h"

class CantileverTestSolver final : public Solver {
   public:
    CantileverTestSolver(wk_ptr<KamiSimulator> _kami);
    void Construct(string settingJsonPath = "") override;
    void SolveNextStep(float dt) override;
    ~CantileverTestSolver();

   private:
    struct CantileverParams {
       public:
        uint32_t numSolvers;
        uint32_t numVertsPerStrands;
        float strandLength;
        float hairInterval;
        glm::vec3 firstHairOrigin;

        CantileverParams() {
            numSolvers = Solver::ModelType::MODEL_TYPE_COUNT - 2;
            numVertsPerStrands = 100;
            strandLength = DEFAULT_HORIZONTAL_HAIR_GEN_PARAM.x;
            hairInterval = DEFAULT_HORIZONTAL_HAIR_GEN_PARAM.y;
            firstHairOrigin = glm::vec3(0, 0, 0);
        }

        static CantileverParams GetDefaultCantileverParams() { return CantileverParams(); };

        CantileverParams(uint32_t _numSolvers, uint32_t _numVertsPerStrands, float _strandLength, float _hairInterval, glm::vec3 _firstHairOrigin);
    };

    // The element names are the same as JSON's key names.
    struct CantileverJsonParams {
        u_int32_t NumSolvers;
        vector<string> SolverModels;  // ∈{"AUGMS","STCOSSERAT"}
        vector<float> WindPeaks;
        vector<float> BendKs;
        vector<float> StretchKs;
        vector<float> Youngs;
        vector<float> ResampleLens;
        vector<bool> UseYoungs;
        vector<bool> SCREnableSDF;

        CantileverJsonParams() { NumSolvers = Solver::ModelType::MODEL_TYPE_COUNT - 2; }
    };

    // The array of kami
    vector<sh_ptr<Solver>> solvers;
    vector<sh_ptr<Hair>> strands;
    CantileverParams params;

    bool LoadSettingJson(string jsonPath, CantileverJsonParams& jsonParams);
    void DownloadHair() override;
    void UploadHair() override;  // This function does nothing.
};

#endif /* CANTILEVER_TEST_SOLVER_H_ */