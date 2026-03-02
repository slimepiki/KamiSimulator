#ifndef HAIR_BODY_MOTION_H_
#define HAIR_BODY_MOTION_H_

#include <fstream>
#include "Kami.h"

#include "Body.h"
#include "Kami.h"
#include "Camera.h"
#include "solvers/Solver.h"
#include "exporters/USCHairsalonExporter.h"
#include "utilities/UnionVec.h"

struct Hair;
class Body;
class Motion;
class Solver;
class KamiFile;
class JsonParser;

class KamiSimulator : public std::enable_shared_from_this<KamiSimulator> {
    struct RunTimingsStamp {
        double simlationTiming;
        double renderTiming;
    };

    // The element names are the same as JSON's key names.
    struct JsonParams {
        string OutputNamePrefix = "test";  // OutputNamePrefix specifies the name of the files that contain simulation results.
                                           //  DO NOT include any extensions and directories in the value
        // KamiSimulator's output format.
        string OutputType = "NOTHING";  // ∈ {"NOTHING", "SEQUENTIAL_OBJ_AND_DATA", "VIDEO_ONLY", "VIDEO_AND_OBJ"}

        // Enable outputting the log file
        bool EnableLog = false;

        // Start consecutive experiments if this value is true. The details are in doc/consec.md.
        bool DoConsecutiveExperiments = false;

        // The path of the JSON file for consecutive experiments, which includes an extension .json. The parameters are described in doc/consec.md.
        string ConsecutiveExperimentsJsonPath = "";

        // The path of the JSON for the solver's parameters that are consistent across the consecutive experiments.
        string ConsecExpBaseJsonPath = "";

        // Hair file's path. If this remains empty, TransplantHair(10, 100), which creates a horizontal hair array, is called.
        string HairPath = "";

        // Body's path. If this remains empty, EmBody(), which creates a box, will be called.
        string BodyPath = "";

        // Body's type. f this remains empty, EmBody(), which creates a box, will be called.
        string BodyType = "";  // ∈ {"HEAD_ONLY, "FULL_BODY", "BOX"}

        // Box's minimal/maximal corner's position. This value is referred to if BodyType is BOX.
        vector<float> BoxMin, BoxMax;

        //(simulation lehgth(s), frame per sec(f/s), internal step (draw step))
        vector<int> SimulationTimings = {2, 60, 5};

        // solver's model
        string SolverModel = "AUG_MASS_SPRING";  // ∈ enum Solver::ModelType

        // The path of the JSON file for the solver that you are going to use. The details are shown in the corresponding solver's header.
        string SolverSettingPath = "";

        // Camera's settings. You must fill all the camera's parameters if you set the camera manually.
        vector<float> CameraPos, CameraLookAt, CameraUp;

        //(num of strands, maximum number of vertices of the strand)
        vector<int> HairSize = {10000, 100};

        void LoadFromJsonParser(JsonParser* jParser);
        void SaveAs(string path);
    };

    struct ConsecutiveJsonParams {
        string ResultNamePrefix = "";

        // You don't have to set this parameter because it's set automatically by referring to the others.
        int NumOfAxis = 1;

        // ∈{"bool", "int", "float", "string"}
        string Axis1Type = "";
        string Axis2Type = "";

        string Axis1Key = "";
        string Axis2Key = "";

        UnionVec Axis1Val;
        UnionVec Axis2Val;

        void LoadFromJsonParser(JsonParser* jParser);
    };

   private:
    sh_ptr<Hair> hair = nullptr;
    sh_ptr<Body> body = nullptr;
    sh_ptr<Motion> motion = nullptr;
    sh_ptr<Solver> solver = nullptr;

    Camera::Setting camera = Camera::BOX_CAMERA_SETTING;

    sh_ptr<KamiFile> logFile = nullptr;

    Kami::KamiSimulatorParams params;

    glm::vec3 glovalPos = glm::vec3(0, 0, 0);
    uint32_t frameCount = 0;

    bool environmentConfirmed = false;
    bool hairGenerated = false;
    bool JsonCameraEnable = false;
    static constexpr uint32_t MAX_FRAME = 99999;
    string solverSettingJsonPath = "";
    string timeStamp = "";

   public:
    KamiSimulator(string outputNamePrefix = "test");
    KamiSimulator(Kami::KamiSimulatorParams initParams);
    ~KamiSimulator();
    // Please include the extension in filePath.
    bool TransplantHair(string filePath, uint32_t maxStrand = 10000, uint32_t maxLength = 100);
    // Generate horizontally straight Hairs in a row
    bool TransplantHair(uint32_t numstrands, uint32_t numVerttsPerStrand, glm::vec3 origin = glm::vec3(0, 0, 0),
                        float hairLength = DEFAULT_HORIZONTAL_HAIR_GEN_PARAM.x, float interval = DEFAULT_HORIZONTAL_HAIR_GEN_PARAM.y);

    // please use Solver::ModelType
    bool PickSolver(int mode);

    // Please include the extension in filePath.
    bool Embody(string filePath, Body::BodyType bType);
    // This constructor generates box.obj to enable the meshPath in the PNMesh as a side effect.
    bool Embody(glm::vec3 boxMin = -Body::DEFAULT_BOX_HALFWIDTH,
                glm::vec3 boxMax = glm::vec3(0, Body::DEFAULT_BOX_HALFWIDTH.y, Body::DEFAULT_BOX_HALFWIDTH.z));
    // Please include the extension in filePath.
    bool Choreograph(string filePath);

    //_drawStepSize determines the minimum unit of the simulation.
    bool SetSimulationTimings(float _simulationLength, uint32_t _framePerSec = 30u, uint32_t _internalStep = 4u);

    void SetOutputType(Kami::OutputType oType);

    void SetSolverSettingJson(string jsonPath);

    bool ConfirmSimulationEnvironment();

    // for test
    void CreateDummy(wk_ptr<Hair> _hair, wk_ptr<Body> _body, Kami::KamiSimulatorParams _params);

    void SetLogFile();
    sh_ptr<KamiFile> GetLogFile();

    bool Simulate();

    // you can save Body's SDF anytime after generating it
    void SaveBodySDF();

    sh_ptr<Hair> GetHair();
    sh_ptr<Body> GetBody();
    sh_ptr<Motion> GetMotion();

    Kami::KamiSimulatorParams GetKamiParams() { return params; };
    float GetInternalStepSize();

    string GenerateFileName(string name);

    void SetEnvironmentFromJson(string jsonPath);

    void ResetAll();

   private:
    RunTimingsStamp SolveNextDrawStep();
    static uint32_t PublishSimulatorID();

    // Output the current frame's .obj and .data
    void OutputHairAndObj();
    // Render the current frame
    double RenderWithMitsuba();
    // Output the result
    void GenerateVideo();

    // The returned string doesn't include any extensions.
    string GenerateCurrentFrameFileName(string name);
    // e.g. frame 1 -> 00001
    string Genereate5DigFrameNum(uint32_t num);
    string GenereateCurrent5DigFrameNum();

    float GetDrawStepSize();
    void ConsecutiveExp(string jsonPath);
};

#endif /* HAIR_BODY_MOTION_H_ */
