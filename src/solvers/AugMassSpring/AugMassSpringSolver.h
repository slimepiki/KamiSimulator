#ifndef AUG_MASS_SPRING_SOLVER_H_
#define AUG_MASS_SPRING_SOLVER_H_

#include <memory>

#include "../../Kami.h"
#include "../Solver.h"
#include "../../KamiSimulator.h"
#include "HairSim.cuh"

class KamiSimulator;
class HairGenerator;
class SimpleObject;
class AMSSolver;

struct HairInfo;
struct HeadInfo;

using std::shared_ptr;

class AugMassSpringSolver : public Solver {
   protected:
    AugMassSpringSolver() {};

   public:
    AugMassSpringSolver(wk_ptr<KamiSimulator> _kami) : Solver(_kami) {};

    // Please precompute and generate auxiliary etc. here.
    void Construct(string settingJsonPath = "") override;
    void UseSDFOrNot(bool yn);

    // dt : step size [ms]
    virtual void SolveNextStep(float dt) override;
    void DownloadHair() override;
    void UploadHair() override;

   private:
    // The element names are the same as JSON's key names.
    struct JsonParams {
        float WindPeak = 3.f;
        float SDFThreshold = 0.013f;
    };
    HairInfo hairInfo;
    HeadInfo headInfo;
    sh_ptr<AMSSolver> amsSolver;
    shared_ptr<HairGenerator> hairGen;
    vector<shared_ptr<SimpleObject>> meshes;

    void GenerateHairAndHeadGrid();
    void LoadSettingJsonContent(string jsonPath);
};

#endif /* AUG_MASS_SPRING_SOLVER_H_ */
