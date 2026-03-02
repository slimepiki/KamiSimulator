#include "FileUtil.h"
#include "Notification.h"
#include <filesystem>
#include <filesystem>

using std::fstream;

bool Kami::FileUtil::IsFileExist(const std::string& filePath) { return std::filesystem::is_regular_file(filePath); }

KamiFile::KamiFile(string _filepath, std::ios_base::openmode mode, bool isOverwriting) {
    filePath = _filepath;

    if ((mode & std::ios_base::out) && Kami::FileUtil::IsFileExist(_filepath) && !isOverwriting) {
        Kami::Notification::Notify("File: " + _filepath + " will be overwritten.");
    }

    if ((mode & std::ios_base::in) && !std::filesystem::exists(filePath)) {
        throw std::runtime_error("The imput file " + filePath + " doesn't exist.");
    }

    fst->open(filePath, mode);
    if (!fst->is_open()) {
        throw std::runtime_error("Failed to open " + filePath + ".");
    }
}

void KamiFile::ReadFromBinary(void* data, uint32_t size) { fst->read((char*)data, size); }

void KamiFile::WriteToBinary(void* data, uint32_t size) { fst->write((char*)data, size); }

void KamiFile::WriteStringWithNoBreak(string str) {
    *fst << str;
    fst->flush();
}
void KamiFile::WriteStringWithBreak(string str) {
    *fst << str << "\n";
    fst->flush();
}

KamiFile::~KamiFile() {
    if (fst->is_open()) {
        fst->flush();
        fst->close();
    }
}

string Kami::FileUtil::GetFileExtension(const std::string& _filepath) {
    size_t extBegin = _filepath.find_last_of('.');
    string ext = _filepath.substr(extBegin + 1, _filepath.size() - extBegin);

    return ext;
}

bool Kami::FileUtil::DeleteDirectory(const std::string& dirPath, bool suppressCaution) {
    if (!std::filesystem::exists(dirPath) && !suppressCaution) {
        Kami::Notification::Caution(__func__, "Invalid path : " + dirPath);
        return false;
    }

    if (remove(dirPath.c_str()) && !suppressCaution) {
        Kami::Notification::Caution(__func__, "Deleting the file faild : " + dirPath);
        return false;
    }
    return true;
}

bool Kami::FileUtil::DeleteFile(const std::string& filePath) {
    if (!std::filesystem::exists(filePath)) {
        Kami::Notification::Caution(__func__, "Invalid path : " + filePath);
        return false;
    }

    if (remove(filePath.c_str())) {
        return false;
    }
    return true;
}

bool Kami::FileUtil::CallCommand(const std::string& command) {
    const auto callCstr = command.c_str();
    return system(callCstr);
}
