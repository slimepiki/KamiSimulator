#include "KamiSimulator.h"

#include <chrono>
#include <filesystem>
#include <regex>

#include "Hair.h"
#include "PNMesh.h"
#include "Motion.h"

#include "utilities/Notification.h"
#include "utilities/FileUtil.h"

#include "solvers/SolverFactory.h"

#include "exporters/MitsubaCurveExporter.h"
#include "exporters/ObjExporter.h"
#include "exporters/USCHairsalonExporter.h"
#include "exporters/SDFParaviewExporter.h"
#include "exporters/CubeObjGenerator.h"
#include "exporters/MitsubaCameraExporter.h"
#include "importers/USCHairsalonImporter.h"

KamiSimulator::KamiSimulator(string outputNamePrefix) {
    // params.SimulatorID = PublishSimulatorID();
    if (outputNamePrefix == "") {
        Kami::Notification::Caution(__func__, "The OutputNamePrefix is empty. This may cause unintended behaviors.");
    }

    params.OutputNamePrefix = outputNamePrefix + "/" + outputNamePrefix;

    // directory creation
    if (!std::filesystem::exists(outputNamePrefix)) {
        std::filesystem::create_directory(outputNamePrefix);
    }
}

KamiSimulator::KamiSimulator(Kami::KamiSimulatorParams initParams) : params(initParams) {
    if (initParams.OutputNamePrefix == "") {
        Kami::Notification::Caution(__func__, "The OutputNamePrefix is empty. This may cause unintended behaviors.");
    }

    // directory creation
    if (!std::filesystem::exists(initParams.OutputNamePrefix)) {
        std::filesystem::create_directory(initParams.OutputNamePrefix);
    }

    params.OutputNamePrefix = initParams.OutputNamePrefix + "/" + initParams.OutputNamePrefix;
    // params.SimulatorID = PublishSimulatorID();

    SetSimulationTimings(initParams.simulationLengthSec, initParams.drawFramePerSec, initParams.numOfInternalStepPerDrawFrame);
}

KamiSimulator::~KamiSimulator() { ResetAll(); }

bool KamiSimulator::TransplantHair(string filePath, uint32_t maxStrand, uint32_t maxLength) {
    if (environmentConfirmed) {
        Kami::Notification::Warn(__func__, "The simulation environment is already confirmed.", logFile);
        return false;
    }

    if (hair) {
        Kami::Notification::Caution(__func__, "Hair data re-importation occurred.");
    }

    hair.reset(new Hair());
    ImportUSCHairSalonToHair(filePath, hair, maxStrand, maxLength);
    Kami::Notification::Notify("hair: " + filePath + " imported.", logFile);
    return true;
}

bool KamiSimulator::TransplantHair(uint32_t numstrands, uint32_t numVertsPerStrand, glm::vec3 origin, float hairLength, float interval) {
    if (environmentConfirmed) {
        Kami::Notification::Warn(__func__, "The simulation environment is already confirmed.", logFile);
        return false;
    }

    if (hair) {
        Kami::Notification::Caution(__func__, "Hair data re-importation occurred.");
    }
    GenerateHorizontalStrandsArrayAsUSCHairsalon(GenerateFileName("horiHair"), numstrands, numVertsPerStrand, origin, hairLength, interval);
    hair.reset(new Hair());

    ImportUSCHairSalonToHair(GenerateFileName("horiHair") + ".data", hair, numstrands, numVertsPerStrand);
    return true;
}

bool KamiSimulator::PickSolver(int solverModel) {
    if (environmentConfirmed) {
        Kami::Notification::Warn(__func__, "The simulation environment is already confirmed.", logFile);
        return false;
    }
    params.modelType = solverModel;
    if (solver) {
        Kami::Notification::Caution(__func__, "Hair model re-defined occurred.");
    }
    if (solverModel == Solver::ModelType::UNDEF_MODEL) {
        Kami::Notification::Warn(__func__,
                                 "\033[31m ModelType::UNDEF_MODEL \033 is not the valid model type.\n"
                                 "Please select the others.",
                                 logFile);
        return false;
    } /* else if (solverModel == ModelType::TEST) {
        solver.reset(new DummySolver(shared_from_this()));
    }*/
    else {
        solver = SolverFactory::CreateSolver((Solver::ModelType)solverModel, shared_from_this());
    }
    return true;
}

bool KamiSimulator::Embody(string filePath, Body::BodyType bType) {
    if (environmentConfirmed) {
        Kami::Notification::Warn(__func__, "The simulation environment is already confirmed.", logFile);
        return false;
    }

    if (body) {
        Kami::Notification::Warn(__func__, "The body data re-importation occurred.", logFile);
    }

    body.reset(new Body(bType));
    body->SetObj(filePath);

    return true;
}

bool KamiSimulator::Embody(glm::vec3 boxMin, glm::vec3 boxMax) {
    if (environmentConfirmed) {
        Kami::Notification::Warn(__func__, "The simulation environment is already confirmed.", logFile);
        return false;
    }

    if (body) {
        Kami::Notification::Warn(__func__, "The body data re-importation occurred.", logFile);
    }

    GenerateCubeObjFile(GenerateFileName("box"), boxMin, boxMax);
    hairGenerated = true;

    body.reset(new Body(Body::BodyType::BOX));
    body->SetObj(GenerateFileName("box") + ".obj");

    return true;
}

bool KamiSimulator::Choreograph(string filePath) {
    if (environmentConfirmed) {
        Kami::Notification::Warn(__func__, "The simulation environment is already confirmed.", logFile);
        return false;
    }
    if (motion) {
        Kami::Notification::Warn(__func__, "The motion data re-importation occurred.", logFile);
    }
    motion.reset(new Motion(filePath));

    return true;
}

bool KamiSimulator::SetSimulationTimings(float _simulationLength, uint32_t _drawFramePerSec, uint32_t _numOfInternalStepPerDrawFrame) {
    if (environmentConfirmed) {
        Kami::Notification::Warn(__func__, "The simulation environment is already confirmed.", logFile);
        return false;
    }

    if (_drawFramePerSec == 0 || _numOfInternalStepPerDrawFrame == 0) {
        Kami::Notification::Warn(__func__,
                                 "_drawFramePerSec or _numOfInternalStepPerDrawFrame is zero!"
                                 "Please give a value other than zero to them.",
                                 logFile);
    }

    params.simulationLengthSec = _simulationLength;
    params.drawFramePerSec = _drawFramePerSec;
    params.numOfInternalStepPerDrawFrame = _numOfInternalStepPerDrawFrame;

    return true;
}

void KamiSimulator::SetOutputType(Kami::OutputType oType) {
    if (environmentConfirmed) {
        Kami::Notification::Warn(__func__, "The simulation environment is already confirmed.", logFile);
        return;
    }
    params.outputType = oType;
}

void KamiSimulator::SetSolverSettingJson(string jsonPath) {
    if (!Kami::FileUtil::IsFileExist(jsonPath)) {
        Kami::Notification::Caution(__func__, "Json: Invalid json path. " + jsonPath);
    } else {
        solverSettingJsonPath = jsonPath;
    }
}

void KamiSimulator::SetLogFile() {
    if (environmentConfirmed) {
        Kami::Notification::Warn(__func__, "The simulation environment is already confirmed.", logFile);
    }
    auto logStart = std::chrono::system_clock::now();

    timeStamp = Kami::Notification::GetCurrentTimeStrForFilename(logStart);

    string filePath = params.OutputNamePrefix + "_log_" + timeStamp + ".kamilog";

    cout << "Logging started. FilePath: " << filePath << endl << endl;

    logFile = make_shared<KamiFile>(filePath, std::ios_base::binary | std::ios_base::out | std::ios_base::trunc);
}

sh_ptr<KamiFile> KamiSimulator::GetLogFile() { return logFile; }

bool KamiSimulator::ConfirmSimulationEnvironment() {
    if (!hair && params.modelType != Solver::ModelType::CANTILEVER) {
        Kami::Notification::Warn(__func__,
                                 "The hair data hasn't been imported.\n"
                                 "Please call \033[31m TransplantHair() \033[0m before calling this.",
                                 logFile);
    }

    if (!body && params.modelType != Solver::ModelType::CANTILEVER) {
        Kami::Notification::Warn(__func__,
                                 "The body data hasn't been imported.\n"
                                 "Please call \033[31m Embody() \033[0m before calling this.",
                                 logFile);
    }

    // if (!motion && params.modelType != CANTILEVER) {
    //     Kami::Notification::Warn(__func__,
    //                              "The motion data hasn't been imported.\n"
    //                              "Please call \033[31m Choreograph() \033[0m before calling this.",
    //                              logFile);
    // }

    if (!solver) {
        Kami::Notification::Warn(__func__,
                                 "The hair modelType hasn't been selected.\n"
                                 "Please call \033[31m PickSolver() \033[0m before calling this.",
                                 logFile);
    }

    if (params.drawFramePerSec == 0 || params.numOfInternalStepPerDrawFrame == 0) {
        Kami::Notification::Warn(__func__,
                                 "_drawFramePerSec or _numOfInternalStepPerDrawFrame is zero!"
                                 "Please give a value other than zero to them.",
                                 logFile);
    }

    if (params.outputType == Kami::OutputType::UNDEF_OUTPUT) {
        Kami::Notification::Warn(__func__,
                                 "The Saving havn't been specified or the invalid value.\n"
                                 "Please call \033[31m SetOutputType() \033[0m before calling this.",
                                 logFile);
    }
    if (JsonCameraEnable) {
        ;
    } else if (body->GetBodyType() == Body::BodyType::BOX || params.modelType == Solver::ModelType::CANTILEVER) {
        camera = Camera::BOX_CAMERA_SETTING;
    } else if (body->GetBodyType() == Body::BodyType::HEAD_ONLY) {
        camera = Camera::HEAD_CAMERA_SETTING;
    } else if (body->GetBodyType() == Body::BodyType::FULLBODY) {
        camera = Camera::FULLBODY_CAMERA_SETTING;
    }

    Kami::Notification::Notify("\nThe environment was confirmed.");
    environmentConfirmed = true;

    body->GenerateSDF();
    body->Confirm();
    Kami::Notification::Notify("\033[1A\033[0JSolver constructing now...");
    solver->Construct(solverSettingJsonPath);
    Kami::Notification::Notify("Construction succeeded!");
    return true;
}

void KamiSimulator::CreateDummy(wk_ptr<Hair> _hair, wk_ptr<Body> _body, Kami::KamiSimulatorParams _params) {
    hair = _hair.lock();
    body = _body.lock();
    params = _params;
    environmentConfirmed = true;
}

bool KamiSimulator::Simulate() {
    RunTimingsStamp sumTime = {0, 0};

    if (!environmentConfirmed) {
        return true;
    }

    Kami::Notification::Notify("\n#####################Simulation info#########################", logFile);
    Kami::Notification::Notify("(Draw Length (s), draw Frame per sec (frame/s), internal step(step)): (" + to_string(params.simulationLengthSec) +
                                   ", " + to_string(params.drawFramePerSec) + ", " + to_string(params.numOfInternalStepPerDrawFrame) + ")",
                               logFile);
    Kami::Notification::Notify(
        "(Draw frame size, Internal step size): (" + to_string(GetDrawStepSize()) + ", " + to_string(GetInternalStepSize()) + ")", logFile);
    auto simStart = std::chrono::system_clock::now();
    Kami::Notification::Notify("The simulation started at " + Kami::Notification::GetCurrentTimeStr(simStart), logFile);
    Kami::Notification::Notify("#########################Solver info##########################", logFile);
    Kami::Notification::Notify(solver->GetInfoString(), logFile);
    Kami::Notification::Notify("################################################################", logFile);
    while (frameCount * GetDrawStepSize() < params.simulationLengthSec && frameCount <= MAX_FRAME) {
        auto elapsed = SolveNextDrawStep();
        sumTime.simlationTiming += elapsed.simlationTiming;
        sumTime.renderTiming += elapsed.renderTiming;
    }
    auto simEnd = std::chrono::system_clock::now();
    Kami::Notification::Notify("The simulation endeded at " + Kami::Notification::GetCurrentTimeStr(simEnd), logFile);
    Kami::Notification::Notify("Total frame is " + to_string(frameCount) + " frames.", logFile);
    Kami::Notification::Notify("Total execute time: " + to_string((int)Kami::Notification::GetElapsedTimeMs(simStart, simEnd)) + "[ms]", logFile);
    Kami::Notification::Notify("Total simulate time: " + to_string((int)sumTime.simlationTiming) + "[ms]", logFile);

    if (params.outputType == Kami::VIDEO_AND_OBJ || params.outputType == Kami::VIDEO_ONLY) {
        Kami::Notification::Notify("Total render time: " + to_string((int)sumTime.renderTiming) + "[ms]", logFile);
        GenerateVideo();
    }

    return true;
}

void KamiSimulator::SaveBodySDF() {
    if (!environmentConfirmed) {
        Kami::Notification::Caution(
            __func__,
            "The environment hasn't be confirmed yet. \n"
            " Please call \033[31m ConfirmSimulationEnvironment() \033[0m before calling this to confirm that the Hair is already specified.",
            logFile);
    } else {
        ExportSDFAsParaViewTSV(params.OutputNamePrefix + "_sdf", body->GetCurrentHeadOrBoxPNMeshPtr());
    }
}

sh_ptr<Hair> KamiSimulator::GetHair() {
    if (!environmentConfirmed) {
        Kami::Notification::Caution(
            __func__,
            "The environment hasn't be confirmed yet. \n"
            " Please call \033[31m ConfirmSimulationEnvironment() \033[0m before calling this to confirm that the Hair is already specified.",
            logFile);
    }
    return hair;
}

sh_ptr<Body> KamiSimulator::GetBody() {
    if (!environmentConfirmed) {
        Kami::Notification::Caution(
            __func__,
            "The environment hasn't be confirmed yet. \n"
            " Please call \033[31m ConfirmSimulationEnvironment() \033[0m before calling this to confirm that the Body is already specified.",
            logFile);
    }
    return body;
}

sh_ptr<Motion> KamiSimulator::GetMotion() {
    if (!environmentConfirmed) {
        Kami::Notification::Caution(
            __func__,
            "The environment hasn't be confirmed yet. \n"
            "Please call \033[31m ConfirmSimulationEnvironment() \033[0m before calling this to confirm that the Motion is already specified.",
            logFile);
    }
    return motion;
}

KamiSimulator::RunTimingsStamp KamiSimulator::SolveNextDrawStep() {
    static const string line = "----------";
    double simTime = 0, renderTime = 0;
    if (!environmentConfirmed) {
        Kami::Notification::Warn(__func__,
                                 "The environment hasn't be confirmed yet. \n"
                                 "Please call \033[31m ConfirmSimulationEnvironment() \033[0m before calling this.",
                                 logFile);
        return {0, 0};
    }

    string frameNo = "frame#" + GenereateCurrent5DigFrameNum();
    auto drawStepSimStart = std::chrono::system_clock::now();

    Kami::Notification::Notify(line + frameNo + line, logFile);
    for (uint32_t i = 0; i < params.numOfInternalStepPerDrawFrame; i++) {
        string substepNo = "substep#" + std::to_string(i);

        auto substepStart = std::chrono::system_clock::now();
        solver->SolveNextStep(GetInternalStepSize());
        auto substepEnd = std::chrono::system_clock::now();

        Kami::Notification::NotifyElapsedTimeMs(substepNo, substepStart, substepEnd, logFile);
    }
    auto drawStepSimEnd = std::chrono::system_clock::now();
    solver->DownloadHair();

    Kami::Notification::NotifyElapsedTimeMs("Draw step's simulation time :" + frameNo, drawStepSimStart, drawStepSimEnd, logFile);
    simTime = Kami::Notification::GetElapsedTimeMs(drawStepSimStart, drawStepSimEnd);

    if (params.outputType != Kami::OutputType::UNDEF_OUTPUT && params.outputType != Kami::OutputType::NOTHING) OutputHairAndObj();

    if (params.outputType == Kami::VIDEO_ONLY || params.outputType == Kami::VIDEO_AND_OBJ) {
        renderTime = RenderWithMitsuba();
    }

    // Deletes .obj and .data if you only want the video.
    if (params.outputType == Kami::VIDEO_ONLY) {
        string hairDataPath = GenerateCurrentFrameFileName("hair") + ".data";
        string bodyDataPath = GenerateCurrentFrameFileName("body") + ".obj";
        Kami::FileUtil::DeleteFile(hairDataPath);
        Kami::FileUtil::DeleteFile(bodyDataPath);
        Kami::Notification::Notify("VIDEO_ONLY: The hair and body are deleted!", logFile);
    }
    string cameraDataPath = GenerateCurrentFrameFileName("camera") + ".tsv";
    Kami::FileUtil::DeleteFile(cameraDataPath);

    Kami::Notification::Notify("", logFile);
    frameCount++;
    return {simTime, renderTime};
}

uint32_t KamiSimulator::PublishSimulatorID() {
    static uint32_t ID = 0;
    return ID++;
}

string KamiSimulator::GenerateFileName(string name) { return params.OutputNamePrefix + "_" + name; }

void KamiSimulator::SetEnvironmentFromJson(string jsonPath) {
    JsonParser jp(jsonPath);

    JsonParams jParams;
    jParams.LoadFromJsonParser(&jp);

    if (jParams.DoConsecutiveExperiments) {
        ConsecutiveExp(jsonPath);
        return;
    }

    // directory creation
    if (!std::filesystem::exists(jParams.OutputNamePrefix)) {
        std::filesystem::create_directory(jParams.OutputNamePrefix);
    }

    params.OutputNamePrefix = jParams.OutputNamePrefix + "/" + jParams.OutputNamePrefix;

    // enabling log file
    if (jParams.EnableLog) {
        SetLogFile();
    }

    // hair construction
    if (jParams.HairPath != "") {
        TransplantHair(jParams.HairPath, jParams.HairSize[0], jParams.HairSize[1]);
    } else {
        TransplantHair(10, 100);  // horizontal hair arraya
    }

    // body construction
    if (jp.CheckKey("BodyType")) {
        auto bTypeStr = jp.GetString("BodyType");
        Body::BodyType bType = Body::BodyType::BOX;

        if (jParams.BodyType == "HEAD_ONLY") {
            bType = Body::BodyType::HEAD_ONLY;
        } else if (jParams.BodyType == "FULLBODY") {
            bType = Body::BodyType::FULLBODY;
        } else if (jParams.BodyType == "BOX") {
            bType = Body::BodyType::BOX;
        } else {
            Kami::Notification::Warn(__func__, "Json: HairBodyMotion: Invalid BodyType (" + jParams.BodyType + ") !");
        }

        if (bType == Body::BodyType::BOX) {
            if (jp.CheckKey("BoxMin") && jp.CheckKey("BoxMax")) {
                if (jParams.BoxMin.size() < 3 || jParams.BoxMax.size() < 3) {
                    Kami::Notification::Caution(__func__,
                                                "Json: HairBodyMotion: Invalid BoxMin or BoxMax. Please check that they have three entries.\n The "
                                                "default Embody() is called for avoiding a crush.");
                    Embody();
                } else {
                    Embody(glm::vec3(jParams.BoxMin[0], jParams.BoxMin[1], jParams.BoxMin[2]),
                           glm::vec3(jParams.BoxMax[0], jParams.BoxMax[1], jParams.BoxMax[2]));
                }
            } else {
                Embody();
            }
        } else {
            if (!jp.CheckKey("BodyPath")) {
                Kami::Notification::Warn(__func__, "Json: HairBodyMotion: Please set " + Kami::Notification::MakeRedString("BodyPath") +
                                                       " if BodyType == \"HEAD_ONLY\" or \"FULLBODY\".");
            }
            if (!Kami::FileUtil::IsFileExist(jParams.BodyPath)) {
                Kami::Notification::Warn(__func__,
                                         "Json: HairBodyMotion: The path " + Kami::Notification::MakeRedString("BodyPath") + " is invalid.");
            }
            Embody(jParams.BodyPath, bType);
        }
    } else {
        Embody();
    }

    // setting output type
    Kami::OutputType oType = Kami::OutputType::VIDEO_ONLY;

    if (jp.CheckKey("OutputType")) {
        if (jParams.OutputType == "NOTHING")
            oType = Kami::OutputType::NOTHING;
        else if (jParams.OutputType == "SEQUENTIAL_OBJ_AND_DATA")
            oType = Kami::OutputType::SEQUENTIAL_OBJ_AND_DATA;
        else if (jParams.OutputType == "VIDEO_AND_OBJ")
            oType = Kami::OutputType::VIDEO_AND_OBJ;
        else if (jParams.OutputType == "VIDEO_ONLY")
            oType = Kami::OutputType::VIDEO_ONLY;
    }

    SetOutputType(oType);

    // setting simulation timings
    glm::ivec3 simTim = glm::ivec3(2, 60, 5);
    if (jp.CheckKey("SimulationTimings")) {
        if (jParams.SimulationTimings.size() < 3) {
            Kami::Notification::Caution(__func__,
                                        "Json: HairBodyMotion: Invalid SimulationTimings. Please check that it has three entries.\n The "
                                        "default values are set for avoiding a crush.");
        } else {
            auto timVec = jParams.SimulationTimings;
            simTim = glm::ivec3(timVec[0], timVec[1], timVec[2]);
        }
    }
    SetSimulationTimings(simTim.x, simTim.y, simTim.z);

    // set a solver
    Solver::ModelType sType = Solver::ModelType::UNDEF_MODEL;
    if (!jp.CheckKey("SolverModel")) {
        Kami::Notification::Warn(__func__, "Json: HairBodyMotion: " + Kami::Notification::MakeRedString("SolverModel") + " must be set.");
    }

    if (jParams.SolverModel == "AUG_MASS_SPRING")
        sType = Solver::ModelType::AUG_MASS_SPRING;
    else if (jParams.SolverModel == "STABLE_COSSERAT_RODS")
        sType = Solver::ModelType::STABLE_COSSERAT_RODS;
    else if (jParams.SolverModel == "CANTILEVER")
        sType = Solver::ModelType::CANTILEVER;
    else if (jParams.SolverModel == "SEP_AUGMS_FOR_TEST")
        sType = Solver::ModelType::SEP_AUGMS_FOR_TEST;

    PickSolver(sType);

    // Solver's setting json
    if (jp.CheckKey("SolverSettingPath")) {
        if (Kami::FileUtil::IsFileExist(jParams.SolverSettingPath)) {
            SetSolverSettingJson(jp.GetString("SolverSettingPath"));
        } else {
            Kami::Notification::Caution(__func__, "Json: HairBodyMotion: Invalid Solver's Setting json path.");
        }
    }

    // camera
    Camera::Setting cameraSetting{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    uint32_t cameraSetCount = 0;
    if (jp.CheckKey("CameraPos")) cameraSetCount++;
    if (jp.CheckKey("CameraLookAt")) cameraSetCount++;
    if (jp.CheckKey("CameraUp")) cameraSetCount++;
    if (cameraSetCount > 0) {
        JsonCameraEnable = true;

        // Check whether the keys exist.
        if (!jp.CheckKey("CameraPos")) {
            Kami::Notification::Caution(__func__, "Json: HairBodyMotion: Key \"CameraPos\" doesn't exist.");
            JsonCameraEnable = false;
        }
        if (!jp.CheckKey("CameraLookAt")) {
            Kami::Notification::Caution(__func__, "Json: HairBodyMotion: Key \"CameraLookAt\" doesn't exist.");
            JsonCameraEnable = false;
        }
        if (!jp.CheckKey("CameraUp")) {
            Kami::Notification::Caution(__func__, "Json: HairBodyMotion: Key \"CameraUp\" doesn't exist.");
            JsonCameraEnable = false;
        }

        if (JsonCameraEnable) {
            // check vectors' size
            if (jParams.CameraPos.size() < 3) {
                Kami::Notification::Caution(
                    __func__,
                    "Json: HairBodyMotion: Invalid \"CameraPos\". Please check that it has three entries.\n The default values "
                    "are set for avoiding a crush.");
                JsonCameraEnable = false;
            }

            if (jParams.CameraLookAt.size() < 3) {
                Kami::Notification::Caution(
                    __func__,
                    "Json: HairBodyMotion: Invalid \"CameraLookAt\". Please check that it has three entries.\n The default values "
                    "are set for avoiding a crush.");
                JsonCameraEnable = false;
            }

            if (jParams.CameraUp.size() < 3) {
                Kami::Notification::Caution(
                    __func__,
                    "Json: HairBodyMotion: Invalid \"CameraUp\". Please check that it has three entries.\n The default values "
                    "are set for avoiding a crush.");
                JsonCameraEnable = false;
            }

            if (JsonCameraEnable) {
                camera.origin = glm::vec3(jParams.CameraPos[0], jParams.CameraPos[1], jParams.CameraPos[2]);
                camera.lookAt = glm::vec3(jParams.CameraLookAt[0], jParams.CameraLookAt[1], jParams.CameraLookAt[2]);
                camera.up = glm::vec3(jParams.CameraUp[0], jParams.CameraUp[1], jParams.CameraUp[2]);
            }
        }
    }

    ConfirmSimulationEnvironment();
}

void KamiSimulator::ResetAll() {
    hair = nullptr;
    body = nullptr;
    motion = nullptr;
    solver = nullptr;
    logFile = nullptr;
    environmentConfirmed = false;
    hairGenerated = false;
    frameCount = 0;
}

string KamiSimulator::GenerateCurrentFrameFileName(string name) { return GenerateFileName(name) + "_" + GenereateCurrent5DigFrameNum(); }

void KamiSimulator::OutputHairAndObj() {
    auto bType = body->GetBodyType();

    Kami::Notification::Notify("", logFile);
    string hairFilename = GenerateCurrentFrameFileName("hair");
    ExportHairAsUSCHairsalon(hairFilename, hair);
    Kami::Notification::Notify("Hair: Saved at " + hairFilename + ".data.", logFile);

    string cameraFilename = GenerateCurrentFrameFileName("camera");
    ExporteMitsubaCameraTSV(cameraFilename, camera);
    Kami::Notification::Notify("Camera: Saved at " + cameraFilename + "tsv.", logFile);

    string bodyFilename = GenerateCurrentFrameFileName("body");
    ExportBodyToObj(bodyFilename, body);

    if (bType == Body::BodyType::HEAD_ONLY)
        Kami::Notification::Notify("Head: Saved at " + bodyFilename + ".obj.", logFile);
    else if (bType == Body::BodyType::FULLBODY)
        Kami::Notification::Notify("Fullbody: Saved at " + bodyFilename + ".obj.", logFile);
    else if (bType == Body::BodyType::BOX)
        Kami::Notification::Notify("Box: Saved at " + bodyFilename + ".obj.", logFile);

    Kami::Notification::Notify("", logFile);
}

double KamiSimulator::RenderWithMitsuba() {
    Kami::Notification::Notify("Rendering: ", logFile);

    string imgPath = GenerateCurrentFrameFileName("img") + ".png";
    string hairDataPath = GenerateCurrentFrameFileName("hair") + ".data";
    string bodyDataPath = GenerateCurrentFrameFileName("body") + ".obj";
    string cameraDataPath = GenerateCurrentFrameFileName("camera") + ".tsv";

    string hairMcurName = GenerateCurrentFrameFileName("mitsuba");
    string hairMcurPath = hairMcurName + ".mcur";

    //  create .mcur (curve)
    ExportMitsubaCurve(hairMcurName, hair);
    Kami::Notification::Notify("Mitsuba curve: Saved at " + hairMcurPath, logFile);

    // render
    auto renderStart = std::chrono::system_clock::now();
    string renderCall = "python3 ../scripts/mitsuba/KamiMitRender.py ";

    renderCall += imgPath;
    renderCall += " ";
    renderCall += hairMcurPath;
    renderCall += " ";
    renderCall += bodyDataPath;
    renderCall += " ";
    renderCall += cameraDataPath;

    Kami::Notification::Notify("Start Rendering...", logFile);
    Kami::FileUtil::CallCommand(renderCall);
    Kami::Notification::Notify("Image: Saved at " + imgPath, logFile);

    auto renderEnd = std::chrono::system_clock::now();

    Kami::Notification::NotifyElapsedTimeMs("Render time", renderStart, renderEnd, logFile);
    Kami::Notification::Notify("", logFile);
    double elapsed = Kami::Notification::GetElapsedTimeMs(renderStart, renderEnd);
    // del mcur
    Kami::FileUtil::DeleteFile(hairMcurPath);
    return elapsed;
}

void KamiSimulator::GenerateVideo() {
    string ffmpegCall = "ffmpeg";
    string imgPath = GenerateFileName("img") + "_\%05d.png";
    auto videoDate = std::chrono::system_clock::now();

    if (timeStamp == "") timeStamp = Kami::Notification::GetCurrentTimeStrForFilename(videoDate);

    string videoPath = GenerateFileName("result") + "_" + to_string(params.drawFramePerSec) + "FPS_" + timeStamp + ".mp4";

    // ffmpeg -r 30 -i image_%03d.png -vcodec libx264 -pix_fmt yuv420p -r 60 out.mp4
    ffmpegCall += " -hide_banner -loglevel error -r ";
    ffmpegCall += to_string(params.drawFramePerSec);
    ffmpegCall += " -i ";
    ffmpegCall += imgPath;
    ffmpegCall += " -vcodec libx264 -pix_fmt yuv420p -r ";
    ffmpegCall += to_string(params.drawFramePerSec);
    ffmpegCall += " ";
    ffmpegCall += videoPath;

    Kami::FileUtil::CallCommand(ffmpegCall);
    Kami::Notification::Notify("The video was generated.", logFile);

    for (uint32_t i = 0; i < frameCount; i++) {
        Kami::FileUtil::DeleteFile(GenerateFileName("img") + "_" + Genereate5DigFrameNum(i) + ".png");
    }
    Kami::Notification::Notify("The sequantial images were deleted.", logFile);

    if (hairGenerated) {
        // GenerateFileName("horiHair") + ".data"
        Kami::FileUtil::DeleteFile(GenerateFileName("horiHair") + ".data");
    }

    if (body->GetBodyType() == Body::BodyType::BOX) {
        Kami::FileUtil::DeleteFile(GenerateFileName("box") + ".obj");
    }
}

string KamiSimulator::Genereate5DigFrameNum(uint32_t num) {
    std::ostringstream oss;
    if (num > MAX_FRAME) {
        return "tooLong";
    }
    oss << std::setw(5) << std::setfill('0') << num;
    return oss.str();
}

string KamiSimulator::GenereateCurrent5DigFrameNum() { return Genereate5DigFrameNum(frameCount); }

float KamiSimulator::GetDrawStepSize() { return 1.0f / params.drawFramePerSec; }

void KamiSimulator::ConsecutiveExp(string jsonPath) {
    JsonParser jParser(jsonPath);
    JsonParams jp;
    jp.LoadFromJsonParser(&jParser);

    Kami::FileUtil::DeleteDirectory(jp.OutputNamePrefix, true);

    jp.DoConsecutiveExperiments = false;
    jp.SolverSettingPath = "ConsecTempSolver.json";

    JsonParser cjpars(jp.ConsecutiveExperimentsJsonPath);
    ConsecutiveJsonParams cjpara;
    cjpara.LoadFromJsonParser(&cjpars);

    uint32_t axis1Size = cjpara.NumOfAxis >= 1 ? cjpara.Axis1Val.GetSize() : 0;
    uint32_t axis2Size = cjpara.NumOfAxis == 2 ? cjpara.Axis2Val.GetSize() : 0;

    for (uint32_t i = 0; i < axis1Size; ++i) {
        std::stringstream axis1message;

        axis1message << "\n#################Consequtive Experiments###################\n";
        axis1message << "Axis1(" << cjpara.Axis1Key << ", ";

        JsonParser solverjp(jp.ConsecExpBaseJsonPath);
        jp.OutputNamePrefix = cjpara.ResultNamePrefix + "_" + cjpara.Axis1Key + "_";

        // set val of axis1
        if (cjpara.Axis1Val.GetType() == UnionVec::BOOL) {
            solverjp.SetVal(cjpara.Axis1Key, cjpara.Axis1Val.GetBool(i), true);
            jp.OutputNamePrefix += to_string(cjpara.Axis1Val.GetBool(i));
            axis1message << "Bool)[" << i << "]: " << cjpara.Axis1Val.GetBool(i);
        } else if (cjpara.Axis1Val.GetType() == UnionVec::INT) {
            solverjp.SetVal(cjpara.Axis1Key, cjpara.Axis1Val.GetInt(i), true);
            jp.OutputNamePrefix += to_string(cjpara.Axis1Val.GetInt(i));
            axis1message << "Int)[" << i << "]: " << cjpara.Axis1Val.GetInt(i);
        } else if (cjpara.Axis1Val.GetType() == UnionVec::FLOAT) {
            solverjp.SetVal(cjpara.Axis1Key, cjpara.Axis1Val.GetFloat(i), true);
            jp.OutputNamePrefix += to_string(cjpara.Axis1Val.GetFloat(i));
            axis1message << "Float)[" << i << "]: " << cjpara.Axis1Val.GetFloat(i);
        } else if (cjpara.Axis1Val.GetType() == UnionVec::STRING) {
            solverjp.SetVal(cjpara.Axis1Key, cjpara.Axis1Val.GetString(i), true);
            jp.OutputNamePrefix += cjpara.Axis1Val.GetString(i);
            axis1message << "String)[" << i << "]: " << cjpara.Axis1Val.GetString(i);
        }

        if (axis2Size == 0) {
            cout << axis1message.str() << endl;
            printf("##########################################################\n");
            solverjp.SaveJson("ConsecTempSolver.json", true);

            jp.OutputNamePrefix = std::regex_replace(jp.OutputNamePrefix, std::regex("\\."), ",");
            jp.SaveAs("ConsecTemp.json");

            ResetAll();
            SetEnvironmentFromJson("ConsecTemp.json");
            Simulate();

            Kami::FileUtil::DeleteFile("ConsecTemp.json");
            Kami::FileUtil::DeleteFile("ConsecTempSolver.json");
        } else {
            string a1str = jp.OutputNamePrefix;
            for (uint32_t j = 0; j < axis2Size; ++j) {
                jp.OutputNamePrefix = a1str;
                jp.OutputNamePrefix += "_" + cjpara.Axis2Key + "_";
                cout << axis1message.str() << endl;
                cout << "Axis2(" << cjpara.Axis2Key << ", ";

                // set val of axis2
                if (cjpara.Axis2Val.GetType() == UnionVec::BOOL) {
                    solverjp.SetVal(cjpara.Axis2Key, cjpara.Axis2Val.GetBool(j), true);
                    jp.OutputNamePrefix += to_string(cjpara.Axis2Val.GetBool(j));
                    printf("Bool)[%u]: %d\n", j, cjpara.Axis2Val.GetBool(j));

                } else if (cjpara.Axis2Val.GetType() == UnionVec::INT) {
                    solverjp.SetVal(cjpara.Axis2Key, cjpara.Axis2Val.GetInt(j), true);
                    jp.OutputNamePrefix += to_string(cjpara.Axis2Val.GetInt(j));
                    printf("Int)[%u]: %d\n", j, cjpara.Axis2Val.GetInt(j));

                } else if (cjpara.Axis2Val.GetType() == UnionVec::FLOAT) {
                    solverjp.SetVal(cjpara.Axis2Key, cjpara.Axis2Val.GetFloat(j), true);
                    jp.OutputNamePrefix += to_string(cjpara.Axis2Val.GetFloat(j));
                    printf("Float)[%u]: %f\n", j, cjpara.Axis2Val.GetFloat(j));

                } else if (cjpara.Axis2Val.GetType() == UnionVec::STRING) {
                    solverjp.SetVal(cjpara.Axis2Key, cjpara.Axis2Val.GetString(j), true);
                    jp.OutputNamePrefix += cjpara.Axis2Val.GetString(j);
                    cout << "String)[" << j << "]: " << cjpara.Axis2Val.GetString(i) << endl;
                }

                solverjp.SaveJson("ConsecTempSolver.json", true);
                jp.OutputNamePrefix = std::regex_replace(jp.OutputNamePrefix, std::regex("\\."), ",");
                jp.SaveAs("ConsecTemp.json");

                ResetAll();
                printf("##########################################################\n");
                SetEnvironmentFromJson("ConsecTemp.json");
                Simulate();

                Kami::FileUtil::DeleteFile("ConsecTemp.json");
                Kami::FileUtil::DeleteFile("ConsecTempSolver.json");
            }
        }
    }
    ResetAll();
}

float KamiSimulator::GetInternalStepSize() { return 1.0f / (params.drawFramePerSec * params.numOfInternalStepPerDrawFrame); }

void KamiSimulator::JsonParams::LoadFromJsonParser(JsonParser* jParser) {
    if (jParser->CheckKey("OutputNamePrefix")) OutputNamePrefix = jParser->GetString("OutputNamePrefix");
    if (jParser->CheckKey("OutputType")) OutputType = jParser->GetString("OutputType");
    if (jParser->CheckKey("EnableLog")) EnableLog = jParser->GetBool("EnableLog");
    if (jParser->CheckKey("DoConsecutiveExperiments")) DoConsecutiveExperiments = jParser->GetBool("DoConsecutiveExperiments");
    if (jParser->CheckKey("ConsecutiveExperimentsJsonPath")) ConsecutiveExperimentsJsonPath = jParser->GetString("ConsecutiveExperimentsJsonPath");
    if (jParser->CheckKey("ConsecExpBaseJsonPath")) ConsecExpBaseJsonPath = jParser->GetString("ConsecExpBaseJsonPath");
    if (jParser->CheckKey("HairPath")) HairPath = jParser->GetString("HairPath");
    if (jParser->CheckKey("BodyPath")) BodyPath = jParser->GetString("BodyPath");
    if (jParser->CheckKey("BodyType")) BodyType = jParser->GetString("BodyType");
    if (jParser->CheckKey("BoxMin")) BoxMin = jParser->GetFloatVec("BoxMin");
    if (jParser->CheckKey("BoxMax")) BoxMax = jParser->GetFloatVec("BoxMax");
    if (jParser->CheckKey("CameraPos")) CameraPos = jParser->GetFloatVec("CameraPos");
    if (jParser->CheckKey("CameraLookAt")) CameraLookAt = jParser->GetFloatVec("CameraLookAt");
    if (jParser->CheckKey("CameraUp")) CameraUp = jParser->GetFloatVec("CameraUp");
    if (jParser->CheckKey("SimulationTimings")) SimulationTimings = jParser->GetIntVec("SimulationTimings");
    if (jParser->CheckKey("SolverModel")) SolverModel = jParser->GetString("SolverModel");
    if (jParser->CheckKey("SolverSettingPath")) SolverSettingPath = jParser->GetString("SolverSettingPath");
    if (jParser->CheckKey("HairSize")) HairSize = jParser->GetIntVec("HairSize");

    return;
}

void KamiSimulator::JsonParams::SaveAs(string path) {
    JsonParser jParser;
    jParser.SetVal("OutputNamePrefix", OutputNamePrefix);
    jParser.SetVal("OutputType", OutputType);
    jParser.SetVal("EnableLog", EnableLog);
    jParser.SetVal("DoConsecutiveExperiments", DoConsecutiveExperiments);
    jParser.SetVal("ConsecutiveExperimentsJsonPath", ConsecutiveExperimentsJsonPath);
    jParser.SetVal("HairPath", HairPath);
    jParser.SetVal("BodyPath", BodyPath);
    jParser.SetVal("BodyType", BodyType);
    jParser.SetVal("BoxMin", BoxMin);
    jParser.SetVal("BoxMax", BoxMax);
    if (CameraPos.size() == 3) jParser.SetVal("CameraPos", CameraPos);
    if (CameraLookAt.size() == 3) jParser.SetVal("CameraLookAt", CameraLookAt);
    if (CameraUp.size() == 3) jParser.SetVal("CameraUp", CameraUp);
    jParser.SetVal("SimulationTimings", SimulationTimings);
    jParser.SetVal("SolverModel", SolverModel);
    jParser.SetVal("SolverSettingPath", SolverSettingPath);
    jParser.SetVal("HairSize", HairSize);

    jParser.SaveJson(path, true);
}

void KamiSimulator::ConsecutiveJsonParams::LoadFromJsonParser(JsonParser* jp) {
    if (jp->CheckKey("ResultNamePrefix")) ResultNamePrefix = jp->GetString("ResultNamePrefix");

    if (jp->CheckKey("Axis1Type")) {
        Axis1Type = jp->GetString("Axis1Type");
        NumOfAxis = 1;

        if (jp->CheckKey("Axis2Type")) {
            Axis2Type = jp->GetString("Axis2Type");
            NumOfAxis = 2;
        }
    }

    if (NumOfAxis != 1 && NumOfAxis != 2) {
        Kami::Notification::Warn(__func__, "Prease set \"Axis1Type\"'s value.\n");
    }

    if (!jp->CheckKey("Axis1Key")) Kami::Notification::Warn(__func__, "Axis1Key is not found\n");
    Axis1Key = jp->GetString("Axis1Key");

    // set Axis1Val
    if (Axis1Type == "bool") {
        Axis1Val.Set(jp->GetBoolVec("Axis1Val"));
    } else if (Axis1Type == "int") {
        Axis1Val.Set(jp->GetIntVec("Axis1Val"));
    } else if (Axis1Type == "float") {
        Axis1Val.Set(jp->GetFloatVec("Axis1Val"));
    } else if (Axis1Type == "string") {
        Axis1Val.Set(jp->GetStringVec("Axis1Val"));
    } else {
        Kami::Notification::Warn(__func__, "Axis1Type is invalid!\n");
    }

    // set Axis2Val if it exists
    if (NumOfAxis == 2) {
        if (!jp->CheckKey("Axis2Key")) Kami::Notification::Warn(__func__, "Axis2Key is not found\n");
        Axis2Key = jp->GetString("Axis2Key");
        if (Axis2Type == "bool") {
            Axis2Val.Set(jp->GetBoolVec("Axis2Val"));
        } else if (Axis2Type == "int") {
            Axis2Val.Set(jp->GetIntVec("Axis2Val"));
        } else if (Axis2Type == "float") {
            Axis2Val.Set(jp->GetFloatVec("Axis2Val"));
        } else if (Axis2Type == "string") {
            Axis2Val.Set(jp->GetStringVec("Axis2Val"));
        } else {
            Kami::Notification::Warn(__func__, "Axis2Type is invalid!\n");
        }
    }
}
