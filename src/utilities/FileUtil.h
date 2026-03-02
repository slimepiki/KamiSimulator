#ifndef FILE_UTIL_H_
#define FILE_UTIL_H_
#include <fstream>

#include "../Kami.h"

namespace Kami::FileUtil {
string GetFileExtension(const std::string& filePath);
bool DeleteFile(const std::string& filePath);
bool DeleteDirectory(const std::string& dirPath, bool suppressCaution = false);
bool IsFileExist(const std::string& filePath);
bool CallCommand(const std::string& command);
}  // namespace Kami::FileUtil

class KamiFile {
   private:
    string filePath;
    sh_ptr<std::fstream> fst = make_unique<std::fstream>();

   public:
    KamiFile(string _filePath, std::ios_base::openmode mode, bool isOverwriting = false);

    void ReadFromBinary(void* data, uint32_t size);
    void WriteToBinary(void* data, uint32_t size);

    void WriteStringWithNoBreak(string str);
    void WriteStringWithBreak(string str);

    sh_ptr<std::fstream> GetFstream() { return fst; }

    ~KamiFile();
};

#endif /* FILE_UTIL_H_ */