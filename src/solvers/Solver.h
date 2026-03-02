#ifndef SOLVER_H_
#define SOLVER_H_

#include "../Kami.h"
#include "../KamiSimulator.h"
#include "../utilities/JsonParser.h"

class KamiSimulator;
struct Hair;
class Body;
class Motion;

class Solver {
   private:
    wk_ptr<KamiSimulator> kami;
    wk_ptr<Hair> hair;
    wk_ptr<Body> body;
    wk_ptr<Motion> motion;
    sh_ptr<Kami::KamiSimulatorParams> params;

   protected:
    Solver() {};
    wk_ptr<Hair> GetHair();
    wk_ptr<Body> GetBody();
    wk_ptr<Motion> GetMotion();

    bool isConstructed = false;

    Kami::KamiSimulatorParams GetKamiParams();
    float GetKamiInternalStepSize();
    string GenerateKamiFilename(string str);

   public:
    enum ModelType {
        // Undefined model
        UNDEF_MODEL,
        // for framework test
        // TEST,
        // for SDF test
        SEP_AUGMS_FOR_TEST,

        // A Mass Spring Model for Hair Simulation(2008)
        //( https://physbam.stanford.edu/~fedkiw/papers/stanford2008-02.pdf )
        // SELLE_MASS_SPRING,

        // DigitalSalon: A Fast Simulator for Dense Hair via Augmented Mass-Spring Model(2024)
        // ( https://repository.kaust.edu.sa/items/b4a5000e-58c8-4b94-aa88-0e2638884b4c )
        AUG_MASS_SPRING,
        // Stable Cosserat Rods(2025)
        //( https://jerryhsu.io/projects/StableCosseratRods/ )
        STABLE_COSSERAT_RODS,
        // cantilever test
        CANTILEVER,

        // ##############################################
        // this element returns the number of ModelType
        // ##############################################
        MODEL_TYPE_COUNT
    };

    Solver(wk_ptr<KamiSimulator> _kami) : kami(_kami) {};
    virtual ~Solver();

    // Please precompute or generate auxiliary things here.
    // use settingJsonPath if you use json setting.
    virtual void Construct(string settingJsonPath = "") = 0;

    // dt : step size [ms]
    // In most cases, Hair's currentPos and currentVert is used as interface between the solver and KamiSimulator.
    virtual void SolveNextStep(float dt) = 0;

    // communicate between the solver and KamiSimulator
    virtual void UploadHair() = 0;

    // communicate between the solver and KamiSimulator
    virtual void DownloadHair() = 0;

    // provide information to KamiSimulator as a string
    virtual string GetInfoString();

    void SubstituteHair(wk_ptr<Hair> _hair);
    void SubstituteBody(wk_ptr<Body> _body);
    void SubstituteMotion(wk_ptr<Motion> _motion);
    void SubstituteKamiimParams(Kami::KamiSimulatorParams _params);
};

#endif /* SOLVER_H_ */