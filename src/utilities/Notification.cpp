#include "Notification.h"

#include <sstream>
#include <iomanip>

#include "../Hair.h"
#include "LinearUtil.h"
#include "FileUtil.h"

using Kami::LinearUtil::EigenVecX3f2DArray;
// Importance
// Warn(invalid manipulation) > Caution(existence of the failed possibility) > Notify(for print something)

void Kami::Notification::Warn(string functionName, string content, sh_ptr<KamiFile> logFile) {
    std::stringstream strSt;
    strSt << "Warning at: \033[31m " + functionName + " \033[0m " << endl;
    strSt << "Warning content: " << content << "\n";
    cerr << strSt.str() << endl;
    if (logFile) {
        logFile->WriteStringWithBreak(strSt.str());
    }
    logFile->~KamiFile();
    throw std::runtime_error(" ");
}

void Kami::Notification::Caution(string functionName, string content, sh_ptr<KamiFile> logFile) {
#ifdef DEBUG_MODE
    std::stringstream strSt;
    strSt << "Caution at: \033[31m " + functionName + " \033[0m " << "\n";
    strSt << "Caution content: " << content << "\n";
    cerr << strSt.str() << endl;
    if (logFile) {
        logFile->WriteStringWithBreak(strSt.str());
    }
#endif
}

void Kami::Notification::Notify(string content, sh_ptr<KamiFile> logFile) {
    std::stringstream strst;
    strst << content;
    cout << strst.str() << endl;
    if (logFile) {
        logFile->WriteStringWithBreak(strst.str());
    }
}

double Kami::Notification::GetElapsedTimeMs(std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

string Kami::Notification::GetCurrentTimeStr(std::chrono::system_clock::time_point time) {
    auto timet = std::chrono::system_clock::to_time_t(time);
    std::tm* lt = std::localtime(&timet);
    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count();
    std::stringstream ss;
    ss << lt->tm_mday;
    ss << "/" << std::setfill('0') << std::right << std::setw(2) << lt->tm_mon + 1;
    ss << "/" << std::setfill('0') << std::right << std::setw(2) << lt->tm_year + 1900;
    ss << " " << std::setfill('0') << std::right << std::setw(2) << lt->tm_hour;
    ss << ":" << std::setfill('0') << std::right << std::setw(2) << lt->tm_min;
    ss << ":" << std::setfill('0') << std::right << std::setw(2) << lt->tm_sec;
    ss << "." << std::setfill('0') << std::right << std::setw(3) << (ms % 1000);

    return ss.str();
}

string Kami::Notification::GetCurrentTimeStrForFilename(std::chrono::system_clock::time_point time) {
    auto timet = std::chrono::system_clock::to_time_t(time);
    std::tm* lt = std::localtime(&timet);
    uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count();
    std::stringstream ss;
    ss << lt->tm_mday;
    ss << "_" << std::setfill('0') << std::right << std::setw(2) << lt->tm_mon + 1;
    ss << "_" << std::setfill('0') << std::right << std::setw(2) << lt->tm_year + 1900;
    ss << "_" << std::setfill('0') << std::right << std::setw(2) << lt->tm_hour;
    ss << "_" << std::setfill('0') << std::right << std::setw(2) << lt->tm_min;
    ss << "_" << std::setfill('0') << std::right << std::setw(2) << lt->tm_sec;
    ss << "_" << std::setfill('0') << std::right << std::setw(3) << (ms % 1000);

    return ss.str();
}

void Kami::Notification::NotifyElapsedTimeMs(string name, std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end,
                                             sh_ptr<KamiFile> logFile) {
    std::stringstream strSt;

    strSt << name << ": " << GetElapsedTimeMs(start, end) << "[ms]";
    cout << strSt.str() << endl;

    if (logFile) {
        logFile->WriteStringWithBreak(strSt.str());
    }
}

void Kami::Notification::PrintHairVertex3D(Hair& hair, uint32_t u, uint32_t v, uint32_t w) {
    auto& vertices = hair.GetCurrentVerticesRef();
    PrintGLMVec3(vertices->GetEntryVal(u, v));
}

void Kami::Notification::PrintHairVertex2D(Hair& hair, uint32_t strand, uint32_t l) {
    auto& vertices = hair.GetCurrentVerticesRef();
    PrintGLMVec3(vertices->GetEntryVal(strand, l));
}

void Kami::Notification::PrintGLMVec3(glm::vec3 vec) { printf("vec3 x: %f, y: %f, z: %f\n", vec.x, vec.y, vec.z); }

void Kami::Notification::PrintGLMMat4(glm::mat4 mat) {
    printf("Mat4x4\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n",  //
           mat[0][0], mat[0][1], mat[0][2], mat[0][3],                          //
           mat[1][0], mat[1][1], mat[1][2], mat[1][3],                          //
           mat[2][0], mat[2][1], mat[2][2], mat[2][3],                          //
           mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
}

string Kami::Notification::GetNumberString(uint32_t i) {
    string ret = to_string(i);
    if (i % 10 == 1)
        ret += "st";
    else if (i % 10 == 1)
        ret += "nd";
    else if (i % 10 == 1)
        ret += "rd";
    else
        ret += "th";
    return ret;
}
