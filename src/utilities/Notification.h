#ifndef KAMI_NOTIFICATION_H_
#define KAMI_NOTIFICATION_H_

#include <chrono>
#include "../Kami.h"

struct Hair;
class KamiFile;

namespace Kami::Notification {
void Warn(string functionName, string content, sh_ptr<KamiFile> logFile = nullptr);

void Caution(string functionName, string content, sh_ptr<KamiFile> logFile = nullptr);

void Notify(string content, sh_ptr<KamiFile> logFile = nullptr);

inline string MakeRedString(string str) { return "\033[31m " + str + "\033[0m "; }

double GetElapsedTimeMs(std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end);

string GetCurrentTimeStr(std::chrono::system_clock::time_point time);
string GetCurrentTimeStrForFilename(std::chrono::system_clock::time_point time);

void NotifyElapsedTimeMs(string name, std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end,
                         sh_ptr<KamiFile> logFile = nullptr);

void PrintHairVertex3D(Hair& hair, uint32_t u, uint32_t v, uint32_t w);
void PrintHairVertex2D(Hair& hair, uint32_t strand, uint32_t l);

void PrintGLMVec3(glm::vec3 vec);
void PrintGLMMat4(glm::mat4 mat);

string GetNumberString(uint32_t i);
}  // namespace Kami::Notification

#endif /* KAMI_NOTIFICATION_H_ */