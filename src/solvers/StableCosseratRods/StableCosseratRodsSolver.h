

#ifndef STABLE_COSSERAT_RODS_SOLVER_H_
#define STABLE_COSSERAT_RODS_SOLVER_H_

#include "../Solver.h"
#include "KittenEngine/includes/modules/Bound.h"
#include "KittenEngine/includes/modules/Rotor.h"
#include "KittenEngine/includes/modules/Dist.h"

#include <set>

// Stable Cosserat Rods (2025) Jerry Hsu et. al.
// ( https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/ )
// I don't use the hair-hair collision routine(KittenGpuLBVH).

class SDFBodyCollisionSolver;

namespace YarnBall {
class Sim;
}

namespace Kami::LinearUtil {
class GLMV3f1DArray;
}

class StableCosseratRodsSolver : public Solver {
    // The internal max step size at the original state is 1e-3;
   public:
    StableCosseratRodsSolver(wk_ptr<KamiSimulator> _kami);
    void Construct(string settingJsonPath = "") override;
    void SolveNextStep(float dt) override;
    ~StableCosseratRodsSolver();

    void UploadHair() override;
    void DownloadHair() override;
    string GetInfoString() override;

   private:
    // The element names are the same as JSON's key names.
    struct JsonParams {
        // options
        string YBJsonPath;
        bool EnableResample;
        bool EnableSDF;
        bool EnableHairHairColl;
        bool UseYoungsModule;

        // simulation parameters
        int NumItr;
        int SagFreeItr;   //
        int SubdivCoeff;  // hair subdivision coefficient. (e.g. simulate 3x segments of initial hair if you select SubdivCoeff = 3.)
        float StretchK;   // stretch stiffness [GPa]?
                          // The default value is 3e3 from Stable Cosserat Rods, Fig. 13(
                          // https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/Stable_Cosserat_Rods-SIGGRAPH25.pdf )
        float BendK;      // bending stiffness [GPa]?
                          // The default value is 6e-4 from Stable Cosserat Rods, Fig. 13(
                          // https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/Stable_Cosserat_Rods-SIGGRAPH25.pdf )

        float YoungsModule;             // Young's module [GPa]. This will used when UseYoungsModule == true;
                                        // The default value is 5 from The Secrets of Beautiful Hair: Why is it Flexible and Elastic? (
                                        // https://www.mdpi.com/2079-9284/6/3/40 ) (isotropic ver).
        float StiffnessDeviationCoeff;  // Bend stiffness's deviation from Theoretical value.
                                        // [1(no deviation),0(zero bending stiffness)]
        float CurveRadius;              // [m]
                            // Yarn radius. Note that this is the minimum radius. The actual radius used in the hair-hair collision is r + 0.5 *
                            // barrierThickness.
                            // The default value, 0.00004[m] is from Interactive Hair Simulation on the GPU using ADMM
                            // (https://dl.acm.org/doi/pdf/10.1145/3588432.3591551 )
        float         //
            Density;  // [kg/m^3] this value relates to mass.
                      // The default value, 1300[kg/m^-3] is from Interactive Hair Simulation on the GPU using ADMM (
                      // https://dl.acm.org/doi/pdf/10.1145/3588432.3591551 )
                      // or
                      // (https://pmc.ncbi.nlm.nih.gov/articles/PMC2872558/#:~:text=At%20full%20hydration%2C%20hair%20can,its%20dry%20weight%20in%20water.&text=The%20density%20of%20dry%20hair,g/cm3%20of%20water.)
        float ResampleLen;    // Length of the resampled segments(important for stability).
        float VelocityDecay;  // YarnBall.h says this is Velocity decay, but cable_work_pattern says this is Linear velocity drag.
        float ExternalForceMultiplier;

        // wind parameters
        float WindPeak;
        float WindYFreq;
        float WindZFreq;
        float WindTimeFreq;
        float WindSharpness;
        JsonParams() {
            YBJsonPath = "";
            EnableResample = true;
            EnableSDF = true;
            EnableHairHairColl = false;
            UseYoungsModule = true;

            NumItr = 8;
            SagFreeItr = 0;
            SubdivCoeff = 3;
            StretchK = 3e3;
            BendK = 6e-4;
            YoungsModule = 5;
            StiffnessDeviationCoeff = 1;
            CurveRadius = 0.00004;
            Density = 1300;
            ResampleLen = 1e-3;
            VelocityDecay = 0.05;
            ExternalForceMultiplier = 1;

            WindPeak = 40.f;
            WindYFreq = 1.f;
            WindZFreq = 0.7f;
            WindTimeFreq = 0.4f;
            WindSharpness = 3.f;
        }
    };
    sh_ptr<YarnBall::Sim> sim = nullptr;
    unq_ptr<SDFBodyCollisionSolver> sdfSolver;
    unq_ptr<Kitten::Dist> simSpeedDist = make_unique<Kitten::Dist>();
    Kitten::Bound<> initialBounds;

    JsonParams jsonParams = JsonParams();

    vector<int> rootVertices;
    // We have to store the related strands because this solver will skip short strands.
    vector<int> strandRefs;

    // bool exportSim = false;
    // float EXPORT_DT = 1 / 30.f;
    // int exportLimit = 2000;
    // bool headlessMode = false;
    // string exportPath = "./frames/frame_";

    int SelectCuda();
    // modificate root positions here by using rootVertices when the Head moving.
    void ModificateRoot();
    sh_ptr<YarnBall::Sim> BuildYarnBallHair(JsonParams jp);
    JsonParams LoadSettingJsonContent(string jsonPath);
    string jDump;
};  // namespace Kami::LinearUtil::GLMV3f1DArray

#endif /* STABLE_COSSERAT_RODS_SOLVER_H_ */
